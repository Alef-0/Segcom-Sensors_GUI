import numpy as np
import cv2 as cv
import math

# Frame dimensions
WIDTH = 800
HEIGHT = 600

MAX_VALUES = 15

# Parameters
# argv[1] = Camera Topic to subscribe


class Graph_radar():
    def __init__(self):
        # Define graph parameters
        self.margin = 50
        self.graph_width = WIDTH - 2 * self.margin
        self.graph_height = HEIGHT - 2 * self.margin
        self.origin_x = WIDTH // 2
        self.origin_y = HEIGHT - self.margin
        self.x_min, self.x_max = -MAX_VALUES, MAX_VALUES
        self.y_min, self.y_max = 0, MAX_VALUES
        self.base_image = self.create_base_image()
        self.base_image = self.create_details()

    def graph_to_pixel(self, x, y):
        pixel_x = int(self.origin_x + (x * self.graph_width) / (self.x_max - self.x_min))
        pixel_y = int(self.origin_y - (y * self.graph_height) / (self.y_max - self.y_min))
        return (pixel_x, pixel_y)
    
    def create_base_image(self):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        # Draw border
        cv.rectangle(frame, (self.margin // 2, self.margin // 2), 
                    (WIDTH - self.margin // 2, HEIGHT - self.margin // 2), (0, 0, 0), 2)

        # Draw X-axis
        cv.line(frame, (self.margin, self.origin_y), (WIDTH - self.margin, self.origin_y), (0, 0, 0), 2)
        # Draw Y-axis
        cv.line(frame, (self.origin_x, HEIGHT - self.margin), (self.origin_x, self.margin), (0, 0, 0), 2)
        
        # Draw grid lines (lighter)
        for x in range(self.x_min, self.x_max + 1):
            if x == 0:
                continue
            pixel_x, _ = self.graph_to_pixel(x, 0)
            cv.line(frame, (pixel_x, self.margin), (pixel_x, HEIGHT - self.margin), (200, 200, 200), 1)

        for y in range(self.y_min, self.y_max + 1):
            if y == 0:
                continue
            _, pixel_y = self.graph_to_pixel(0, y)
            cv.line(frame, (self.margin, pixel_y), (WIDTH - self.margin, pixel_y), (200, 200, 200), 1)

        # Draw arrows
        # X-axis arrow
        cv.arrowedLine(frame, (WIDTH - self.margin - 10, self.origin_y), 
                        (WIDTH - self.margin, self.origin_y), (0, 0, 0), 2, tipLength=0.02)
        # Y-axis arrow
        cv.arrowedLine(frame, (self.origin_x, self.margin + 10), 
                        (self.origin_x, self.margin), (0, 0, 0), 2, tipLength=0.02)

        # Draw tick marks and labels for X-axis
        for x in range(self.x_min, self.x_max + 1, 5):
            if x == 0:
                continue  # Skip origin
            pixel_x, pixel_y = self.graph_to_pixel(x, 0)
            cv.line(frame, (pixel_x, pixel_y - 5), (pixel_x, pixel_y + 5), (0, 0, 0), 1)
            cv.putText(frame, str(x), (pixel_x - 10, pixel_y + 20), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw tick marks and labels for Y-axis
        for y in range(self.y_min, self.y_max + 1, 5):
            if y == 0:
                continue  # Skip origin
            pixel_x, pixel_y = self.graph_to_pixel(0, y)
            cv.line(frame, (pixel_x - 5, pixel_y), (pixel_x + 5, pixel_y), (0, 0, 0), 1)
            cv.putText(frame, str(y), (pixel_x - 25, pixel_y + 5), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw origin label
        cv.putText(frame, "0", (self.origin_x - 15, self.origin_y + 20), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw axis labels
        cv.putText(frame, "X", (WIDTH - self.margin + 5, self.origin_y + 5), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv.putText(frame, "Y", (self.origin_x - 5, self.margin - 5), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display the frame
        return frame

    def create_details(self):
        new_img = self.base_image.copy()
        center = self.graph_to_pixel(0,0)

        angle_rad = math.radians(30)
        slope = math.tan(angle_rad)

        # Compute endpoints for ±60° diagonals
        end_right = self.graph_to_pixel(self.x_max, self.x_max * slope)
        end_left  = self.graph_to_pixel(self.x_min, -self.x_min * slope)
        cv.line(new_img, center, end_left, (0, 0, 0), 1)
        cv.line(new_img, center, end_right, (0, 0, 0), 1)

        # Draw the semicircules
        for i in range(1, self.y_max + 1):
            x,y = self.graph_to_pixel(i,i)
            cv.ellipse(new_img, center, (x - center[0], center[1] - y), 0, 270 - 60, 270 + 60, (0,0,0), 1)

        return new_img

    def show_points(self, x_group, y_group, colors):
        new_img = self.base_image.copy()
        for x, y, c in zip(x_group, y_group, colors):
            value = self.graph_to_pixel(x,y)
            cv.circle(new_img, value, 4, c, -1)
            
        cv.imshow("RADAR", new_img)
        cv.waitKey(1)

    def close(self):
        cv.destroyAllWindows()
        cv.waitKey(1)
