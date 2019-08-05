import cv2 as cv
import sys
import numpy as np
#python multi_object_matching.py E:\matching\query_image.jpg E:\matching\template.jpg
def multi_object_matching(query_img_path,template_path):
    img=cv.imread(query_img_path)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #template=gray[78:90,118:140]
    template=cv.imread(template_path,0)
    #cv.imwrite('template.jpg',template)
    res=cv.matchTemplate(gray,template,cv.TM_SQDIFF_NORMED)
    min_val,max_val,min_loc,max_loc=cv.minMaxLoc(res)
    min_thresh=(min_val+1e-6)*310
    #min_thresh = (min_val + 1e-6) * 103000
    match_locations=np.where(res<=min_thresh)
    w,h=template.shape[::-1]
    result=[]
    for(x,y) in zip(match_locations[1],match_locations[0]):
        cv.rectangle(img,(x,y),(x+w,y+h),[0,0,255],2)
        result.append([x,y,x+w,y+h])
    #cv.imwrite('match_result.jpg',img)
    cv.imshow('',img)
    cv.waitKey()
    return result
if __name__=='__main__':
    query_image_path=sys.argv[1]
    template_path=sys.argv[2]
    result=multi_object_matching(query_image_path,template_path)
    print(result)