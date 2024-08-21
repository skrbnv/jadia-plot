### Plot add-on for Jadia diarization package

# Install:
You need [jadia](https://github.com/skrbnv/jadia) installed before using this package
`pip install jadia-plot`

### Usage
```python
# plot segments
plot.plot_segments(
    pred=segments,
    ground_truth=reference,
    filename=SEGMENTS_IMAGE_FILENAME,
)
# or predictions (+segments)
plot.plot_predictions(
    predictions=predictions,
    segments=segments,
    filename=PREDICTIONS_IMAGE_FILENAME,
    ground_truth=reference,
)
```
Look into `eval.ipynb` [here](https://github.com/skrbnv/jadia) notebook for plotting, metrics etc. 

