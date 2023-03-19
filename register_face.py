import os
import cv2
import argparse
def register_face(input_file_path = "",
                  classname = "",
                  database_path="./ImageFace"):
    try:
        cv2.imread(input_file_path)
        cv2.imwrite(database_path+"/"+classname+".jpg")
        print("entry done")
    except:
        print("invalid image file")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", help="Image path of the person with visible face")
    parser.add_argument("classname", help="Name of the person")
    parser.add_argument("-db", help="path to the database where all the other faces are present")

    args = parser.parse_args()
    db="./ImageFace"
    if not args.db is None:
        db = args.db 
    

    register_face(input_file_path=args.input_file_path ,
                  classname=args.classname,
                  database_path=db)

