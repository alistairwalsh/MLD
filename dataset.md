# Workshop dataset

The data for this workshop is from a real research project into honeybee behaviour that was solved with machine learning.

## Classifying Honeybee Tags

To explore honeybee behaviour bees were individually tagged with reflective markers. They were then filmed using an infrared light for 24 hours a day for over two weeks. Three different types of tag were used:

Tag Number             |  Pattern   | Details
:-------------------------:|:-------------------------: | :-------------------------:
1  |  Rectangle | 100 bees that were the control group
2  |  Circle    | 100 bees that were treated with caffeine
3  | Blank      | Single queen in the colony received this tag

This is what the tags looked like before we added them to the colony:

Control Tags             |  Treatment Tags
:-------------------------:|:-------------------------:
![control tags](images/tag1.jpg)  |  ![Treatment Tags](images/tag2.jpg)

Queen Tag             |
:-------------------------:|
![](images/queen.jpg)  |

This is how the tags appeared in the video we filmed over the course of the experiment

Experiment Footage             |
:-------------------------:|
![](images/beehive.png)  |


Ultimately the aim of this experiment was to examine honeybee behaviour by:

1. Identifying locations of bees in individual video frames.
2. Determining whether the located bees were in the control or treatment groups based on the image of the tag on their back.
3. Tracking the location of the bees over time.

We're interested in distinguishing the three tagged groups of bees in the footage from this experiment. There are far too many videos and bees to be able to do this manually, so we'll try to train a machine learning algorithm to perform this classification task for us. 

For this workshop we will focus on step 2: identifying the tags on the back of the bee. The dataset consists of 730 24x24 pixel images cropped from video frames like the above. Each image has been labelled with the type of tag visible. Our job is to automatically identify what the tag is from the content of the image itself.


