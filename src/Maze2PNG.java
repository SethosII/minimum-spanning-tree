import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Maze2PNG {
	static public void main(String args[]) throws Exception {
		System.out.println("Starting ...");
		if (args.length == 2) {
			try {
				// read the maze file created from MST program with -m option
				BufferedReader br = new BufferedReader(new FileReader(args[0]));
				String mazeString = "";
				String line = "";
				while ((line = br.readLine()) != null) {
					if (line.charAt(0) == ' ' || line.charAt(0) == '|'
							|| line.charAt(0) == '+' || line.charAt(0) == '-') {
						// only use lines containing the maze
						mazeString += line + "\n";
					}
				}
				br.close();

				String[] mazeArray = mazeString.split("\n");

				// +2 for padding
				int width = mazeArray[0].length() + 2;
				int height = mazeArray.length + 2;

				// format: 8-bit rgba integer pixels
				BufferedImage mazeImage = new BufferedImage(width, height,
						BufferedImage.TYPE_INT_ARGB);
				for (int row = 0; row < height; row++) {
					for (int column = 0; column < width; column++) {
						if (row == 0 || column == 0 || row == height - 1
								|| column == width - 1) {
							// black padding
							mazeImage.setRGB(column, row, Color.BLACK.getRGB());
						} else {
							// actual mace
							if (mazeArray[row - 1].charAt(column - 1) == ' ') {
								// space white
								mazeImage.setRGB(column, row,
										Color.BLACK.getRGB());
							} else {
								// all other black
								mazeImage.setRGB(column, row,
										Color.WHITE.getRGB());
							}
						}
					}
				}

				ImageIO.write(mazeImage, "PNG", new File(args[1]));
				System.out.println("Finished ...");
			} catch (IOException e) {
				e.printStackTrace();
			}
		} else {
			System.err
					.println("\nYou need to pass two arguments.\n"
							+ "First argument is the text file, e. g. /home/user/maze.txt.\n"
							+ "Second argument is the PNG file to store the image, e. g. /home/user/maze.png.\n");
		}
		System.out.println("Exiting ...");
	}
}