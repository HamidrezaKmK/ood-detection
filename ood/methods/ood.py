# TODO: remove this file after fully refactoring and accounting for both
# tensorboard and wandb logging

"""
This file contains code related to the Laplace approximation used for OOD detection.
"""
import torch
import typing as th
import dypy as dy
from torch.utils import tensorboard

def local_optimization(
    x: torch.Tensor,
    likelihood_model: torch.nn.Module,
    optimization_steps: int,
    optimizer: th.Union[torch.optim.Optimizer, str],
    optimizer_args: th.Optional[th.Dict[str, th.Any]] = None,
    optimization_objective: th.Literal["nll", "grad_norm"] = "nll",
    tensorboard_writer: th.Optional[tensorboard.SummaryWriter] = None,
    intermediate_image_log_frequency: th.Optional[int] = None,
    pick_best: bool = True,
):
    """
    This function performs a local optimization on the input x to the likelihood model.

    The intention is to find a point that is close to x and has a high likelihood which is
    peaked. However, plugging in a tensorboard_writer will also tell you how the likelihood
    function behaves close to x. This is useful for getting insights into the first and second
    order behavior of the likelihood function.

    Args:
        x (torch.Tensor): The input
        likelihood_model (torch.nn.Module): The likelihood parametric model
        optimization_steps (int): Number of steps you want the optimization to run
        optimizer (th.Union[torch.optim.Optimizer, str]):
            The class name of the optimizer to use as a string or as the class itself
        optimizer_args (th.Optional[th.Dict[str, th.Any]], optional):
            The arguments for the optimizer to be instantiated.
            Defaults to None which means the empty dictionary.
        optimization_objective:
            The objective is either the negative log likelihood (-model.log_prob)
            or the norm of the gradient of the nll, i.e., || grad -model.log_prob ||
            Defaults to "nll".
        tensorboard_writer (th.Optional[tensorboard.SummaryWriter], optional):
            You can pass a tensorboard summary writer to this one as well that summarizes the following:
            1. The distance from origin of the image
            2. The beginning image itself
            3. The image it ends up at.
            4. The gradient norms of nll
            5. The nll itself
            Defaults to None, which means that none of these will be logged.
        pick_best (bool, optional):
            If set to True, the model picks the intermediate input that produces
            the smallest objective.
            Defaults to True.
    Returns:
        x0: Manipulates x so that x0 is the point that is peaked in terms of the model likelihood.
    """
    # add the original image to the tensorboard writer
    if tensorboard_writer is not None:
        tensorboard_writer.add_image("original_image", x, 0)
        tensorboard_writer.flush()

    # turn off all the gradients for likelihood_model
    for param in likelihood_model.parameters():
        param.requires_grad_(False)

    def forward_func(x):
        x_ = x.unsqueeze(0).to(likelihood_model.device)
        return -likelihood_model.log_prob(x_).squeeze(0)

    x0 = x.clone().detach().requires_grad_(True)
    optimizer_args = optimizer_args or {}

    # change x to x0 by optimizing the input according to the objective
    optimizer = (
        optimizer([x0], **optimizer_args)
        if isinstance(optimizer, torch.optim.Optimizer)
        else dy.eval(optimizer)([x0], **optimizer_args)
    )

    # the objective is either the negative log likelihood or the gradient norm
    if optimization_objective == "nll":
        obj_func = lambda x: forward_func(x)
    elif optimization_objective == "grad_norm":
        obj_func = lambda x: torch.norm(torch.func.vjp(forward_func, x)[0])
    else:
        raise Exception("optimization_objective must be specified!\n" "Please choose between 'nll' and 'grad_norm'")

    # do the optimization and keep the place where the gradient
    # is the smallest for the point to be a peak
    best_x0 = None
    best_obj = None

    for _ in range(optimization_steps):
        if isinstance(optimizer, torch.optim.LBFGS):
            # Do a second-order optimization with closure
            def closure():
                optimizer.zero_grad()
                loss = obj_func(x0)
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            # Do a simple first order optimization
            optimizer.zero_grad()
            loss = obj_func(x0)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        nll = forward_func(x0)
        nll.backward()
        nll_grad_norm = torch.norm(x0.grad)
        optimizer.zero_grad()

        if pick_best:
            with torch.no_grad():
                t = obj_func(x0)
                if best_obj is None or t < best_obj:
                    best_obj = t
                    best_x0 = x0.clone()

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("||x - x0||", torch.norm(x0 - x), _)
            tensorboard_writer.add_scalar("nll_grad_norm", nll_grad_norm, _)
            tensorboard_writer.add_scalar("nll", nll, _)
            if intermediate_image_log_frequency is not None and (_ + 1) % intermediate_image_log_frequency == 0:
                tensorboard_writer.add_image("intermediate_image", x0, _)
            with torch.no_grad():
                tensorboard_writer.add_scalar("objective", obj_func(x0), _)

    if pick_best:
        x0 = best_x0

    if tensorboard_writer is not None:
        # add x and x0 ass images to the tensorboard writer
        tensorboard_writer.add_image("proxy_image (peaked in terms of likelihood)", x0, 0)
        tensorboard_writer.flush()

    return x0.detach().cpu()


def laplace_score(
    x: torch.Tensor,
    likelihood_model: torch.nn.Module,
    x0: th.Optional[torch.Tensor] = None,
    score_type: th.Literal["log_det", "largest_eigval(s)", "gaussian_diff"] = "log_det",
    score_args: th.Optional[th.Dict[str, th.Any]] = None,
    eps: float = 1e-6,
):
    """
    This score is used for OOD detection on a point 'x'. For a single input x to the
    likelihood generative model calculates the score according to the Laplace approximation.

    It is assumed that x0 is the closest peak to x, (if it is not given, then x is just assumed
    to be a peak itself)

    (Full documentation in the PDF)

    Args:
        x (torch.Tensor): The point we want to determine whether it is OOD or not.
        likelihood_model (torch.nn.Module): The likelihood model.
        x0 (th.Optional[torch.Tensor], optional):
            Defaults to None, which in that case, x is considered as x0 instead.
        score_type: str
            Different types of scores are considered:
            1. 'log_det':
                The log determinant of the Hessian of the log likelihood at x0.
                Since the calculated Hessian is not necessarily positive definite,
                we round up every eigenvalue that is smaller than eps to eps.

            2. 'largest_eigval(s)':
                The largest eigenvalue of the Hessian of the log likelihood at x0.
                If we want to consider more eigenvalues, then we can set the score_args['num_eigvals'].
                This will simply calculate the sum of the logarithm of these eigenvalues.

            3. 'gaussian_diff':
                The difference between the unnormalized density (i.e., the log-likelihood itself) at
                'x' and the normalized density at 'x' which is represented by a Gaussian with mean 'x0'
                The covariance matrix of the Gaussian is the inverse of the Hessian of the log-likelihood

        score_args (th.Optional[th.Dict[str, th.Any]], optional):
            Different arguments for the different score types.

        eps (float, optional):
            The epsilon term used for error correction. Defaults to 1e-6.

    Returns:
        float: The score of the point x that can be used for OOD detection.
    """
    with torch.no_grad():

        def nll(x):
            x_ = x.unsqueeze(0).to(likelihood_model.device)
            return -likelihood_model.log_prob(x_).squeeze(0)

        # calculate the Hessian w.r.t x0
        hess = torch.func.hessian(nll)(x0)

        # Hess has weired dimensions now, we turn it into a 2D matrix
        # by reshaping only the first half.
        # For example, if the shape of hess is (2, 2, 2, 2), then we reshape it to (4, 4)
        # For images of size (1, 28, 28), the shape of hess is (1, 28, 28, 1, 28, 28)
        # and we reshape it to (784, 784)
        hess = hess.squeeze()
        first_half = 1
        for i, dim in enumerate(hess.shape):
            if i < len(hess.shape) // 2:
                first_half *= dim
        hess = hess.reshape(first_half, -1).detach().cpu()

        if score_type == "log_det":
            # get all the eigenvalues of the hessian
            eigvals, eigvectors = torch.linalg.eigh(hess)

            # round up the eigenvalues that are smaller than eps
            eigvals[eigvals < eps] = eps

            # print something if hess has nans
            if torch.isnan(hess).any():
                raise Exception("Hessian has nans")

            score = torch.sum(torch.log(eigvals))
        elif score_type == "largest_eigval(s)":
            raise NotImplementedError("largest_eigval(s) is not implemented yet")
        elif score_type == "gaussian_diff":
            raise NotImplementedError("gaussian_diff is not implemented yet")
        else:
            raise Exception(
                "score_type must be specified!\n" "choose between 'log_det', 'largest_eigval(s)', and 'gaussian_diff'"
            )

    return score
