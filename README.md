# People Counting from Video Using YOLOv5

This assessment demonstrates how to count the number of people in a video using the YOLOv5 object detection model. It includes functionality to download a YouTube video, trim it to a specified segment, and count the people detected in that segment. The assessment also provides basic consistency checks to validate the people counting results.

## Features

- Download a video from YouTube.
- Trim the video to a specified time range.
- Detect and count people in the video using YOLOv5.
- Perform basic consistency checks on the results.

## Setup

### Prerequisites

- Python 3.7 or later
- `pip` for package management

### Install Required Packages

Install the required Python packages using the following command:

```
python3 -m venv .venv
source .venv/bin/activate
pip -r requirements.txt
```

Download the required weights to this repository:

```
https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?pli=1
```

### Run the code:

```
python3 estimate_people.py
```

## Results

I found the results to be lacking. The method of splitting the video into different time windows ended up not being an efficient way to validate the reuslts. This may be because the drone the video was taken with does not maintain a consistent height throughout the video, or angle within the air. This causes problems with the detection of people in the video, as people closer to the y-axis, or where x=0, are more likely to be seen as non-people. This is because people in this area of the video are much smaller, seen as closer to points in space than actual people, causing issues with detection.

A more robust method of validation is necessary here, such as ground-truth comparison, but due to the time constraints of the assessment, this was not performed, and instead the time-window method was implemented. This method would have consisted with a manual validation of the model: Given a particular frame of the video, we can determine the "ground truth" of the frame, and then cross-reference this number to the detected number of people within the particular frame.

Another alternative of validation is cross-referencing with different models. This would have been more timely and efficient, but due to the nature of the problem, another model could not be found whose dataset that it was trained on was similar to the data given. This might be resolved with training a model myself that could detect people at this height, but again, given the time constraints of the assessment, I deemed this route to be too costly.

In short, I _personally_ believe the model does well with the given weights and the specific time window that was provided in the assessment. However, this is nullified by the cross-referencing methodology used.
