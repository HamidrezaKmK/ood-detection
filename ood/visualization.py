"""
This code is intended for visualization purposes, mainly on weights and biases.
All of the particular visualization implementations can be found here.
"""
import numpy as np
import typing as th
import wandb

def visualize_histogram(
    scores: np.ndarray,
    bincount: int = 10,
    reject_outliers: th.Optional[float] = None,
    logger = None,
    x_label: str = 'scores',
    y_label: str = 'density',
    title: str = 'Histogram of the scores',
):
    """
    The generic histogram in weights and biases does not give
    us the capability to manually adjust the binwidth and limits
    our representation capabilities.
    
    In addition to that, if we want to use plotly and matplotlib,
    we cannot overlay all the runs which is crucial in our framework.
    
    Therefore, here we implement a custom histogram implementation.
    We create a table and log the table to have the exact values for
    the histogram later on. Also, we create a set of lines that represent
    the histogram density. Although we are plotting lines in W&B, we are
    actually interpreting them as histograms.
    
    Args:
        scores: The scores that we want to visualize. 
        bincount: The number of bins used in the histogram.
        reject_outliers: The percentage of outliers that we want to reject from the
                    beginning and end. If set to None, no outlier rejection happens.
        logger: The weights and biases logger.
        x_label: The label of the x-axis.
        y_label: The label of the y-axis.
        title: The title of the histogram.
    Returns: None
        It visualizes the scores in W&B.
    """
    # sort all_scores 
    all_scores = np.sort(scores)
    
    # reject the first and last quantiles of the scores for rejecting outliers
    if reject_outliers is not None:
        L = int(reject_outliers * len(all_scores))
        R = int((1 - reject_outliers) * len(all_scores))
        all_scores = all_scores[L: R]
    
    # create a density histogram out of all_scores
    # and store it as a line plot in (x_axis, density)
    hist, bin_edges = np.histogram(all_scores, bins=bincount, density=True)
    density = hist / np.sum(hist)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # get the average distance between two consecutive centers
    avg_dist = np.mean(np.diff(centers))
    # add two points to the left and right of the histogram
    # to make sure that the plot is not cut off
    centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
    density = np.concatenate([[0], density, [0]])
    
    data = [[x, y] for x, y in zip(centers, density)]
    table = wandb.Table(data=data, columns = [x_label, y_label])
    dict_to_log = {f"histogram/{title}": wandb.plot.line(table, x_label, y_label, title=title)}
    
    if logger is not None:
        logger.log(dict_to_log)
    else:
        wandb.log(dict_to_log)

def visualize_scatterplots(
    scores: np.ndarray,
    column_names: th.Optional[th.List[str]] = None,
    title: str = 'Score-Scatterplot',
):
    """
    A set of scores with their corresponding column names are given.
    This would be visualized in a scatterplot in W&B. If we have 
    'k' columns, then we would have 'k choose 2' scatterplots. With each of the
    scatterplots being visualized in W&B showing the representation of each row 
    corresponding to those two metrics.
    
    Args:
        scores (np.ndarray): The scores that we want to visualize.
        column_names (th.Optional[th.List[str]], optional): The name of the columns. Defaults to score-i
        title (str, optional): The title of the scatterplots, defaults to 'Score-Scatterplot'.
    """
    data = []
    column_names = column_names or [f"score-{i}" for i in range(scores.shape[1])]
    for i in range(scores.shape[0]):
        row = []
        for j in range(scores.shape[1]):
            row.append(scores[i, j])
        data.append(row)
        
        x, y = scores[i, 0], scores[i, 1]
        data.append([x, y])
    
    table = wandb.Table(data=data, columns = column_names)
    
    for i in range(len(column_names)):
        for j in range(i+1,len(column_names)):
            wandb.log(
                {
                    f"scatter/{title}-{column_names[i]}-vs-{column_names[j]}": wandb.plot.scatter(table, column_names[i], column_names[j], title=f"{column_names[i]} and {column_names[j]}")
                }
            )

def visualize_trends(
    scores: np.ndarray,
    t_values: np.ndarray,
    reference_scores: th.Optional[np.ndarray] = None,
    with_std: bool = False,
    title: str = 'scores',
    x_label: str = 't-values',
    y_label: str = 'scores',
):
    """
    This function visualizes the trends in the scores. Scores is interpreted as an
    aggregation of scores.shape[0] trends. Each row represents a trend and the t_values
    is a 1D array of size scores.shape[1] representing the t-values of each trend.
    
    If reference_scores is not None, then we would visualize the reference scores as well.
    
    The visualization is done in W&B, where the average of each column is calculated and
    visualized as a line plot. Also, if the with_std is set to True, then the standard deviation
    on the positive side and negative side is also visualized.
    
    Args:
        scores (np.ndarray): An ndarray where each row represents the scores in a trend.
        t_values (np.ndarray): A monotonically increasing array of t-values in a trend.
        reference_scores (th.Optional[np.ndarray], optional): The reference trend that is sometimes used.
        with_std (bool, optional): If set to true, then the std of the trends are also visualized.
        title (str, optional): The title of the plots. Defaults to 'scores'.
        x_label (str, optional): The x-axis name of the trend. Defaults to 't-values'.
        y_label (str, optional): The y-axis name of the trend. Defaults to 'scores'.
    """
    
    mean_scores = []
    mean_minus_std = []
    mean_plus_std = []
    mean_reference_scores = []
    mean_minus_std_reference_scores = []
    mean_plus_std_reference_scores = []
    
    for _, i in zip(t_values, range(scores.shape[1])):
        scores_ = scores[:, i]
        avg_scores = np.nanmean(scores_)
        scores_ = np.where(np.isnan(scores_), avg_scores, scores_)
        
        upper_scores = scores_[scores_ >= avg_scores]
        lower_scores = scores_[scores_ <= avg_scores]
        
        mean_scores.append(avg_scores)
        mean_minus_std.append(avg_scores - np.std(lower_scores))
        mean_plus_std.append(avg_scores + np.std(upper_scores))
        
        if reference_scores is not None:
            reference_scores_ = reference_scores[:, i]
            avg_scores = np.nanmean(reference_scores_)
            reference_scores_ = np.where(np.isnan(reference_scores_), avg_scores, reference_scores_)
            
            upper_scores = reference_scores_[reference_scores_ >= avg_scores]
            lower_scores = reference_scores_[reference_scores_ <= avg_scores]
            mean_reference_scores.append(avg_scores)
            mean_minus_std_reference_scores.append(avg_scores - np.std(lower_scores))
            mean_plus_std_reference_scores.append(avg_scores + np.std(upper_scores))
    
    if with_std:
        ys = [mean_scores, mean_minus_std, mean_plus_std]
        keys = [f"{y_label}-mean", f"{y_label}-std", f"{y_label}+std"]
        if reference_scores is not None:
            ys = [
                mean_reference_scores, 
                mean_minus_std_reference_scores, 
                mean_plus_std_reference_scores
            ] + ys
            keys = [
                f"ref-{y_label}-mean", 
                f"ref-{y_label}-std", 
                f"ref-{y_label}+std"
            ] + keys
        
        wandb.log({
            f"trend/{title}": wandb.plot.line_series(
                xs = t_values,
                ys = ys,
                keys = keys,
                title = title,
                xname = x_label,
            )
        })
    else:
        if reference_scores is not None:
            ys = [mean_reference_scores, mean_scores]
            keys = [f"ref-{y_label}-mean", f"{y_label}-mean"]
            wandb.log({
                f"trend/{title}": wandb.plot.line_series(
                    xs = t_values,
                    ys = ys,
                    keys = keys,
                    title = title,
                    xname = x_label,
                )
            })
        else:
            
            # if no reference or no STD is given, there is no reason to
            # plot line series, we can simply plot a line plot
            table = wandb.Table(data = [[x, y] for x, y in zip(t_values, mean_scores)], columns = [x_label, y_label])
            wandb.log({
                f"trend/{title}": wandb.plot.line(
                    table,
                    x_label,
                    y_label,
                    title=title,
                )
            })