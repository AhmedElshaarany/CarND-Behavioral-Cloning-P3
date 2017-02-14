import csv
import numpy as np
import random
import cv2

# open driving log csv file
f = open("driving_log.csv")

# create csv file reader
csv_f = csv.reader(f)

# define a new ouptput csv file 
ofile  = open('Augmented_driving_log.csv', "w")
writer = csv.writer(ofile, delimiter=',')
writer.writerow(["image","steering"])

# define the steering compenstation for the left and right images
steering_compensation = 0.2


i = 0
count = 0
for row in csv_f:
    # if angle outside the range -0.1 to 0.1, add center image, add left and right images after compensating
    if(i == 1 and (float(row[3]) >= 0.1 or float(row[3]) <= -0.1)):
        row1 = [row[0], float(row[3])]
        writer.writerow(row1)
        row2 = [row[1].strip(), float(row[3])+steering_compensation]
        row3 = [row[2].strip(), float(row[3])-steering_compensation]
        writer.writerow(row2)
        writer.writerow(row3)
        
        
	# if it is a sharp angle, flip the image and the steering angle and add to csv file
        if( (float(row[3]) > 0.2)  or (float(row[3]) < -0.2)) :
            for i in range(3):
                image_name = row[i].strip()
                img = cv2.imread(image_name)
                img = cv2.flip(img,1)
                new_image_name = image_name[:4]+"flipped_"+image_name[4:]
                cv2.imwrite(new_image_name,img)
                if(i==0):
                    rowflipped = [new_image_name, -float(row[3])]
                elif(i==1):
                    rowflipped = [new_image_name, -float(row[3])+steering_compensation]
                else:
                    rowflipped = [new_image_name, -float(row[3])-steering_compensation]
                writer.writerow(rowflipped)

    # if angle inside range -0.1 to 0.1, choose 700 images randomly, add center image, add left and right images after compensating
    elif( i == 1 and random.randint(0,1)==1 and count <= 700):
        row1 = [row[0], float(row[3])]
        row2 = [row[1].strip(), float(row[3])+steering_compensation]
        row3 = [row[2].strip(), float(row[3])-steering_compensation]
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)        
        count += 1
        
    i = 1

    
# close input and output csvfiles
f.close()
ofile.close()

