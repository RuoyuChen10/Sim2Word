from mtcnn import MTCNN
import cv2
img = cv2.cvtColor(cv2.imread("test.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
face = detector.detect_faces(img)[0]

#draw box
box = face["box"]
I = cv2.rectangle(img, (box[0],box[1]),(box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
#draw points
left_eye = face["keypoints"]["left_eye"]
right_eye = face["keypoints"]["right_eye"]
nose = face["keypoints"]["nose"]
mouth_left = face["keypoints"]["mouth_left"]
mouth_right = face["keypoints"]["mouth_right"]
points_list = [(left_eye[0], left_eye[1]),
               (right_eye[0], right_eye[1]),
               (nose[0], nose[1]),
               (mouth_left[0], mouth_left[1]),
               (mouth_right[0], mouth_right[1])]
for point in points_list:
	cv2.circle(I, point, 1, (255, 0, 0), 4)
#show
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
cv2.imwrite('./result.jpg',I)
cv2.imshow('result', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
