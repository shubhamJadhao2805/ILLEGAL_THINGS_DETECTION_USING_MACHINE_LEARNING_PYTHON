import cv2
import numpy as np
import PIL.Image
import coremltools
import operator
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
Height = 227 # use the correct input image height
Width = 227# use the correct input image width

print("Select your option :")
print("1.For Illegal Things  ")
print("2.For Defect Detection")
print("Q.To terminate the process")
mode = int(input())
if mode == 1:
    model = coremltools.models.MLModel("Hackthon.mlmodel")
    print("Process of detecting Illegal Things are started...")

elif mode == 2:
    model = coremltools.models.MLModel("Defect.mlmodel")
    print("Process of detecting Defected Things are started...")

cap = cv2.VideoCapture(0)
resize_to=(Width, Height)
while True:
    ret, frame = cam.read()
    cv2.imshow("test",frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    frame = PIL.Image.fromarray(frame)
    img = frame.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    out_dict = model.predict({'image': img})
    # print(out_dict)
    prediction = out_dict.get("classLabelProbs")
    print(prediction)
    # print(out_dict.get("classLabel"))

#    predKnif = out_dict["knif"]
#    preGuns = out_dictp["GUNS"]
#    preDrinks = out_dict["drinks"]
#    if predKnif > preGuns and preGuns > preDrinks:
#        print("Knif")
#        print(predKnif)
#    elif preGuns > predKqnif and predKnif > preDrinks:
#        print("GUNS")
#        print(preGuns)
#    elif preDrinks>predKnif and predKnif > preGuns:
#        print("Drinks")
#        print(preDrinks)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()
