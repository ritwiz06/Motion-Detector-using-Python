import cv2, time

first_frame = None


video = cv2.VideoCapture(0)  #0 for webcam, 1, 2, 3- for external camera, video file path for pre made Video, DSHOW-port of camera


while True:

    check, frame = video.read() #first frame capturing
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0) #removes noise and increase accuracy, (21, 21) glussian blur kernel

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame =cv2.absdiff(first_frame, gray)

    thresh_delta = cv2.threshold(delta_frame, 30,  255, cv2.THRESH_BINARY)[1]
    thresh_frame_smooth = cv2.dilate(thresh_delta, None, iterations=2)
    #contours tuple
    (cnts,_) = cv2.findContours(thresh_frame_smooth.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #retrieve external, chain approx simple (approximation method to retriving contours)

    for contour in cnts:
        if cv2.contourArea(contour) <1000:  #less than 1000 pixels
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame",delta_frame)

    cv2.imshow("Smooth Threshold Frame", thresh_frame_smooth)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    print(gray)
    print(delta_frame)

    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
