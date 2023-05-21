# ChartRecognizer
Half-finished project of charts recognization from "Kaggle: Benetech - Making Graphs Accessible".  
**Get more information on [Benetech - Making Graphs Accessible](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview/description)**

**Here's a brief explanation about my codes.**
1. I use yolo to recognize different component in the picture.
2. Then manually process these information to fix x-axis and y-axis and the area of main chart.
3. Depending on different type of charts, I cropped the specific part of the picture, and put it into another model.
4. To process 4 types of charts, I used 4 model (manual or by yolo) to process the cropped images.

**Generally, these are not successful codes. It's not robust enough to face a variety of images.  
But I hope these codes can give some help or inspiration to someone in the future.**
