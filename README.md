# String-Art
Pass in your jpg file to processImage, this implementation uses a greedy algorithm with a highlight favoring heuristic to choose the darkest line. <br>
Then it removes that line based on the xiaoline algorithms built in brightness function.

In the future those brightness values can be updated to favor pixels in the center (or somewhere else). Performance can also be improved. but with 4000 lines and 800 hooks, 
this is a good example of what is capable:

<img width="279" alt="image" src="https://github.com/Tay-Day/String-Art/assets/89946561/87c5ceef-cde9-4a58-a8ee-837e635a31be">
