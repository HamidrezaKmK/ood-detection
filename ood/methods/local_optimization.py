from .base_method import OODBaseMethod
import torch
import typing as th
import dypy as dy
import wandb


class OODLocalOptimization(OODBaseMethod):
    """
    This class of methods does a local optimization on OOD samples
    in terms of the input.
    
    Showing some properties such as Negative Log Likelihood (NLL) and
    the gradient of NLL with respect to the input will give us some
    insights into the behavior of the likelihood model.
    
    The intention is to find a point that is close to x and has a high likelihood which is
    peaked.
    """

    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        optimization_steps: int,
        optimizer: th.Union[torch.optim.Optimizer, str],
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        logger: th.Optional[th.Any] = None,
        optimizer_args: th.Optional[th.Dict[str, th.Any]] = None,
        optimization_objective: th.Literal["nll", "grad_norm"] = "nll",
        intermediate_image_log_frequency: th.Optional[int] = None,
        pick_best: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(x=x, x_batch=x_batch, likelihood_model=likelihood_model, logger=logger, **kwargs)
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.optimization_objective = optimization_objective
        self.intermediate_image_log_frequency = intermediate_image_log_frequency
        self.pick_best = pick_best

    def run(self):
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
        # TODO: a bit of refactoring is needed here
        likelihood_model = self.likelihood_model
        x = self.x
        optimization_steps = self.optimization_steps
        optimizer = self.optimizer
        optimizer_args = self.optimizer_args
        optimization_objective = self.optimization_objective
        intermediate_image_log_frequency = self.intermediate_image_log_frequency
        pick_best = self.pick_best

        # add the original image to the tensorboard writer
        if self.logger is not None:
            self.logger.log(
                {"optimization/original_image": [wandb.Image(x, caption="original image we start off with")]}
            )

        # turn off all the gradients for likelihood_model
        for param in self.likelihood_model.parameters():
            param.requires_grad_(False)

        def forward_func(x):
            x_ = x.unsqueeze(0).to(likelihood_model.device)
            return -likelihood_model.log_prob(x_).squeeze(0)

        x0 = x.clone().detach()
        x0.requires_grad = True
        
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
            raise Exception(
                "optimization_objective must be specified!\n" "Please choose between 'nll' and 'grad_norm'"
            )

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

                loss = optimizer.step(closure)
            else:
                # Do a simple first order optimization
                optimizer.zero_grad()
                loss = obj_func(x0)
                loss.backward()
                optimizer.step()
            
            x1 = x0.clone().detach()
            x1.requires_grad = True
            
            nll = forward_func(x1)
            nll.backward()
            nll_grad_norm = torch.norm(x1.grad)
            
            if pick_best:
                with torch.no_grad():
                    t = obj_func(x0)
                    if best_obj is None or t < best_obj:
                        best_obj = t
                        best_x0 = x0.clone()

            if self.logger is not None:
                with torch.no_grad():
                    self.logger.log(
                        {
                            "nll_grad_norm": nll_grad_norm,
                            "nll": nll,
                            "||x - x0||": torch.norm(x0 - x),
                            "objective": obj_func(x0),
                        }
                    )
                if intermediate_image_log_frequency is not None and (_ + 1) % intermediate_image_log_frequency == 0:
                    self.logger.log(
                        {
                            "optimization/intermediate_image": [
                                wandb.Image(x0, caption="The image in the optimization process")
                            ]
                        }
                    )

        if pick_best:
            x0 = best_x0

        if self.logger is not None:
            # add x and x0 images to the tensorboard writer
            self.logger.log(
                {
                    "optimization/final_image": [wandb.Image(x0, caption="The image we end up with")],
                }
            )
        return x0.detach().cpu()
