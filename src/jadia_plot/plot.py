import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from jadia import Segments, cost_matrix
from jadia.hungarian import optimize

TICKS_FONT_SIZE = 6


class KeepOrder:
    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        return index


def format_func(value, tick_number):
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes:02}:{seconds:02}"


def int_to_color(integer, max_val=40, cmap_name="gist_rainbow"):
    cmap = cm.get_cmap(cmap_name)
    return cmap(integer / max_val)


def match_speakers(pred: Segments, ground_truth: Segments):
    gtspk = ground_truth.speakers()
    cmatrix = cost_matrix(pred, ground_truth)
    row_index, col_index = optimize(-cmatrix)
    match = {i: j for i, j in zip(col_index, row_index)}
    for i in range(len(gtspk)):
        if i not in match.keys():
            match[i] = len(match)
    return match


def plot_predictions(
    filename: str,
    predictions: np.ndarray,
    segments: Segments | None = None,
    segments2: Segments | None = None,
    ground_truth: Segments = None,
    hop: int = 160,
    sr: int = 16000,
) -> None:
    """
    Draws predictions into chart
    """
    plt.figure(figsize=(50, 2))
    ax = plt.gca()
    x = np.arange(len(predictions)) * hop / sr
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    plt.xticks(np.arange(min(x), max(x) + 1, (max(x) - min(x)) / 50))
    maxcolors = predictions.shape[1]

    if ground_truth is not None:
        if segments is not None:
            reorder = match_speakers(segments, ground_truth)
        else:
            reorder = KeepOrder()

        speakers_gt = ground_truth.speakers()
        ax.text(
            1, 0.48, "Ground truth (rectangles in background)", fontsize=6, ha="left"
        )
        for s in ground_truth:
            ax.add_patch(
                Rectangle(
                    (s.start, 0),
                    s.end - s.start,
                    1,
                    linewidth=1,
                    facecolor=int_to_color(
                        reorder[speakers_gt.index(s.speaker)], maxcolors
                    ),
                    alpha=0.2,
                )
            )

    ax.text(-1.5, 0.1, "Predictions (curves)", fontsize=6, ha="right")
    for s in range(predictions.shape[1]):
        y = predictions[:, s]
        plt.scatter(
            np.arange(len(predictions)) * hop / sr,
            y,
            c=np.array(int_to_color(s, maxcolors)),
            alpha=0.5,
            s=0.2,
        )
    if segments is not None:
        ax.text(-1.5, 0.98, "Segments", fontsize=6, ha="right")
        speakers = segments.speakers()
        for segment in segments:
            ax.add_patch(
                Rectangle(
                    (segment.start, 1),
                    segment.end - segment.start,
                    0.02,
                    linewidth=1,
                    facecolor=int_to_color(speakers.index(segment.speaker), maxcolors),
                    alpha=1,
                )
            )

    if segments2 is not None and segments is not None:
        ax.text(-1.5, 0.48, "Segments set no.2", fontsize=6, ha="right")
        reorder = match_speakers(segments, segments2)
        speakers = segments2.speakers()
        for segment in segments2:
            ax.add_patch(
                Rectangle(
                    (segment.start, 0.5),
                    segment.end - segment.start,
                    0.02,
                    linewidth=1,
                    facecolor=int_to_color(
                        reorder[speakers.index(segment.speaker)], maxcolors
                    ),
                    alpha=1,
                )
            )

    plt.savefig(filename, dpi=300)
    plt.close()


def plot_segments(filename: str, pred: Segments, ground_truth: Segments | None = None):
    plt.figure(figsize=(50, 2))
    colormap = plt.get_cmap("tab20")
    colors = [colormap(i) for i in range(20)]
    ax = plt.gca()
    maxx = int(max([el.end for el in pred]) + 1)
    if ground_truth is not None:
        maxx = max(maxx, int(max([el.end for el in ground_truth]) + 1))
    x = list(range(maxx))
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.xticks(np.arange(min(x), max(x) + 1, (max(x) - min(x)) / 50))

    if ground_truth is not None:
        reorder = match_speakers(pred, ground_truth)
        speakers_gt = ground_truth.speakers()
        if len(speakers_gt) > 20:
            raise Exception("Too many speakers, unable to process more than 20")

    speakers = pred.speakers()

    if ground_truth is not None:
        ax.text(-1.5, 0.03, "Ground truth", fontsize=6, ha="right")
        for segment in ground_truth:
            ax.add_patch(
                Rectangle(
                    (segment.start, 0),
                    segment.end - segment.start,
                    0.1,
                    linewidth=1,
                    facecolor=colors[reorder[speakers_gt.index(segment.speaker)]],
                    alpha=1,
                )
            )

    ax.text(-1.5, 0.48, "Predicted", fontsize=6, ha="right")
    for segment in pred:
        ax.add_patch(
            Rectangle(
                (segment.start, 0.45),
                segment.end - segment.start,
                0.1,
                linewidth=1,
                facecolor=colors[speakers.index(segment.speaker)],
                alpha=1,
            )
        )

    plt.savefig(filename, dpi=300)
    plt.close()


def unique_in_order(sequence):
    seen = set()
    unique_list = []
    for item in sequence:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list
