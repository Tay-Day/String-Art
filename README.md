# String-Art
A lightweight program for replicating string art online.
## To Use:
### Upload Image to String-Art folder
You can put it elsewhere just be sure to specify its path relative to the string-Art folder in your call

### Call imgProcessor with the command line arguments [filepath for image] [number of strings] [number of hooks] [Line Strength] [Avoid White Dampener]

Line Strength is a value from 0 - 255 that determines how powerful a string is on your image. In practice small values make the strings fixate on darker areas because they remove less black from the image when drawn. Sticking to around 0-100 for the default string width and using more strings works well.

Avoid White Dampener is a value that refers to the algorithms tendancy to avoid white pixels, sometimes to its detriment, so a larger value will punish the algorithm less for moving over white pixels, and smaller values [1-20] usually find a happy medium.

### EXAMPLE COMMAND LINE CALL:
python imgProcessor.py portrait.jpg 2000 150 40 4

This implementation uses a greedy algorithm with a highlight favoring heuristic to choose the darkest line. <br>
Then it removes that line based on the xiaoline algorithms built in brightness function.

In the future those brightness values can be updated to favor pixels in the center (or somewhere else). Performance can also be improved. But with 4000 lines and 800 hooks, 
this is a good example of what is capable:

<img width="279" alt="image" src="https://github.com/Tay-Day/String-Art/assets/89946561/87c5ceef-cde9-4a58-a8ee-837e635a31be">
