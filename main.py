import cv2

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel operator
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Compute the magnitude of gradients
gradient_mag = cv2.magnitude(sobel_x, sobel_y)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Gradient Magnitude', gradient_mag)
cv2.waitKey(0)
cv2.destroyAllWindows()
