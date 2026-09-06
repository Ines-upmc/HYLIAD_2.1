import cv2
import dlib
import face_recognition
import parse_data
from PIL import Image, ImageDraw
import subprocess
from datetime import datetime

#print(cv2.__version__)
#print(dlib.__version__)
#print(face_recognition.__version__)


#In ordering to recognize your personal face you require to add a personal photo of your face with your phone and add it with the name and put it on the directory.

#images/samples/gerard.jpg

#capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

cap = webcam_video_stream

#Load the database
name, birthday, age, profession, mac, height, weight, eye, sex, ethnicity, file_name, biography = parse_data.parse_csv("database_sample.csv")

#load the sample images and get the 128 face embeddings from them
h_image = [face_recognition.load_image_file('images/samples/'+file_name[i]) for i in range(len(file_name))]
#h_image = [face_recognition.load_image_file('images/samples_students/'+file_name[i]) for i in range(len(file_name))]

h_face_encodings = [face_recognition.face_encodings(h_image[i])[0] for i in range(len(h_image))]



known_face_encodings = h_face_encodings
known_face_names = name
known_face_info = [str(age[i])+", "+ profession[i] for i in range(len(age))]

birthday = ["BD "+birthday[i].replace('.','').replace('/','.') for i in range(len(birthday))]
height = ["HET "+str(height[i])+" CM" for i in range(len(height))]
weight = ["WGT "+str(weight[i]) for i in range(len(weight))]
eye = ["EYE "+eye[i] for i in range(len(eye))]
sex = ["SEX "+sex[i] for i in range(len(sex))]
mac = ["MAC "+mac[i] for i in range(len(mac))]
ethnicity = ["ETH "+ethnicity[i] for i in range(len(ethnicity))]
#biography = ["BUREAU RECORD- "+biography[i] for i in range(len(biography))]

#initialize the array variable to hold all face locations, encodings and names
all_face_locations = []
all_face_encodings = []
all_face_names = []

#loop through every frame in the video

while True:
    ret,current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')

    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)


    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location

        #change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4

        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

        #string to hold the label
        name_of_person = 'UNKNOWN'
        info_of_person = '-ERROR'
        name_i = '-ERROR'
        birthday_i = '-ERROR'
        age_i = '-ERROR'
        profession_i = '-ERROR'
        mac_i = '-ERROR'
        height_i = '-ERROR'
        weight_i = '-ERROR'
        eye_i = '-ERROR'
        sex_i = '-ERROR'
        ethnicity_i = '-ERROR'
        biography_i = '-ERROR'

        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            info_of_person = known_face_info[first_match_index]
            name_i = name[first_match_index]
            birthday_i = birthday[first_match_index]
            age_i = age[first_match_index]
            profession_i = profession[first_match_index]
            mac_i = mac[first_match_index]
            height_i = height[first_match_index]
            weight_i = weight[first_match_index]
            eye_i = eye[first_match_index]
            sex_i = sex[first_match_index]
            ethnicity_i = ethnicity[first_match_index]
            biography_i = biography[first_match_index]


            pil_image = Image.fromarray(current_frame)
            # Create a Pillow ImageDraw Draw instance to draw with
            draw = ImageDraw.Draw(pil_image)

            name_ii = '_'.join(name_i.split(' '))
            now_ = datetime.now()
            dt_string = now_.strftime("%d%m%Y-%Hh")
            
            # Display the resulting image
            pil_image.save("images/stream_images/"+name_ii+"_streaming.jpg")
            subprocess.Popen("cp images/stream_images/"+name_ii+"_streaming.jpg images/tmp_images/"+name_ii+"_"+dt_string+".jpg",shell=True)
           
            
            
            
            
        def minimal_rectangle():
            #draw minimal rectangle around the face
            imge = current_frame
            start_point = (left_pos,top_pos)
            end_point = (right_pos,bottom_pos)
            color_frame = (192,192,192)
            thickness = 1
            k = 130
            #cv2.rectangle(imge, start_point, end_point, color_frame, thickness)
            #cv2.rectangle(imge, (left_pos+70,top_pos), (right_pos-70,bottom_pos), color_frame, cv2.FILLED)
        
            cv2.line(imge, (left_pos,top_pos), (right_pos-k, top_pos), color_frame, thickness)
            cv2.line(imge, (right_pos,top_pos), (left_pos+k, top_pos), color_frame, thickness)
            cv2.line(imge, (left_pos,bottom_pos), (right_pos-k, bottom_pos), color_frame, thickness)
            cv2.line(imge, (right_pos,bottom_pos), (left_pos+k, bottom_pos), color_frame, thickness)
        
        
            cv2.line(imge, (left_pos,top_pos), (left_pos, bottom_pos-k), color_frame, thickness)
            cv2.line(imge, (left_pos,bottom_pos), (left_pos, top_pos+k), color_frame, thickness)
            cv2.line(imge, (right_pos,top_pos), (right_pos, bottom_pos-k), color_frame, thickness)
            cv2.line(imge, (right_pos,bottom_pos), (right_pos, top_pos+k), color_frame, thickness)
        
            #display the name as text in the image
            font1 = cv2.FONT_HERSHEY_DUPLEX
            font2 = cv2.FONT_HERSHEY_TRIPLEX
        
            cv2.putText(current_frame, name_of_person, (left_pos,top_pos-25), font1, 0.55, color_frame,1)
            cv2.putText(current_frame, info_of_person, (left_pos,top_pos-8), font1, 0.5, color_frame,1)
            return
        
        def extended_rectangle():
            #draw extended information rectangle around the face
            imge = current_frame
            start_point = (left_pos,top_pos)
            end_point = (right_pos,bottom_pos)
            #color_frame = (192,192,192)
            color_frame = (0,0,0)
            thickness = 1
            k = 130
           

            # Display the name as text in the image
            font1 = cv2.FONT_HERSHEY_DUPLEX
            font2 = cv2.FONT_HERSHEY_TRIPLEX
            
            cv2.line(imge, (left_pos,top_pos), (right_pos-k, top_pos), color_frame, thickness)
            cv2.line(imge, (right_pos,top_pos), (left_pos+k, top_pos), color_frame, thickness)
            cv2.line(imge, (left_pos,bottom_pos), (right_pos-k, bottom_pos), color_frame, thickness)
            cv2.line(imge, (right_pos,bottom_pos), (left_pos+k, bottom_pos), color_frame, thickness)
        
        
            cv2.line(imge, (left_pos,top_pos), (left_pos, bottom_pos-k), color_frame, thickness)
            cv2.line(imge, (left_pos,bottom_pos), (left_pos, top_pos+k), color_frame, thickness)
            cv2.line(imge, (right_pos,top_pos), (right_pos, bottom_pos-k), color_frame, thickness)
            cv2.line(imge, (right_pos,bottom_pos), (right_pos, top_pos+k), color_frame, thickness)

            ###########################################################################
            # First vertical line
            cv2.line(imge, (right_pos,top_pos), (right_pos+230, top_pos), color_frame, thickness)
            cv2.line(imge, (right_pos+18,top_pos+25), (right_pos+230, top_pos+25), color_frame, thickness)
            # First vertical line divisions
            cv2.line(imge, (right_pos+81,top_pos), (right_pos+81, top_pos+25), color_frame, thickness)
            cv2.line(imge, (right_pos+147,top_pos), (right_pos+147, top_pos+25), color_frame, thickness)
            cv2.line(imge, (right_pos+194,top_pos), (right_pos+194, top_pos+25), color_frame, thickness)

            # First vertical line text  
            cv2.putText(current_frame, birthday_i, (right_pos+18,top_pos+15), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, height_i, (right_pos+85,top_pos+15), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, weight_i, (right_pos+151,top_pos+15), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, eye_i, (right_pos+198,top_pos+15), font1, 0.25, color_frame,1)
            
            ##########################################################################
            # Second vertical line
            cv2.line(imge, (right_pos+18,top_pos+50), (right_pos+230, top_pos+50), color_frame, thickness)
            
            # Second vertical line divisions
            cv2.line(imge, (right_pos+55,top_pos+25), (right_pos+55, top_pos+50), color_frame, thickness)
            cv2.line(imge, (right_pos+167,top_pos+25), (right_pos+167, top_pos+50), color_frame, thickness)

            # Second vertical line text  
            cv2.putText(current_frame, sex_i, (right_pos+18,top_pos+39), font1, 0.25, color_frame,1)

            cv2.putText(current_frame, mac_i, (right_pos+61,top_pos+41), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, ethnicity_i, (right_pos+175,top_pos+41), font1, 0.25, color_frame,1)

            #Arrow
            cv2.line(imge, (right_pos+18,top_pos+60), (right_pos+23, top_pos+67), color_frame, thickness)
            cv2.line(imge, (right_pos+23,top_pos+67), (right_pos+28, top_pos+60), color_frame, thickness)

            #Record text
            
            cv2.putText(current_frame, biography_i[0:47], (right_pos+18,top_pos+83), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47:47*2], (right_pos+18,top_pos+93), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*2:47*3], (right_pos+18,top_pos+103), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*3:47*4], (right_pos+18,top_pos+113), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*4:47*5], (right_pos+18,top_pos+123), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*5:47*6], (right_pos+18,top_pos+143), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*6:47*7], (right_pos+18,top_pos+153), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*7:47*8], (right_pos+18,top_pos+163), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*8:47*9], (right_pos+18,top_pos+173), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*9:47*10], (right_pos+18,top_pos+193), font1, 0.25, color_frame,1)
            cv2.putText(current_frame, biography_i[47*10:47*11], (right_pos+18,top_pos+203), font1, 0.25, color_frame,1)
            
            ##########################################################################
            # Top left info text
            cv2.putText(current_frame, name_of_person, (left_pos,top_pos-25), font1, 0.55, color_frame,1)
            cv2.putText(current_frame, info_of_person, (left_pos,top_pos-8), font1, 0.5, color_frame,1)
            
            
            return

        extended_rectangle()
        #minimal_rectangle()

    
    #display the video
    cv2.imshow("Webcam Video", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()
