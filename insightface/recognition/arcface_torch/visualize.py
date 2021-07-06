import matplotlib.image as image
import numpy as np
import cv2

def visualize(dataset,start_idx,end_idx,num):
    img_vertical = None
    result = None
    frame_end_idx = end_idx
    #에러
    if start_idx >= end_idx : 
        print("올바른 end index를 입력하세요")

    #빈 공간으로 메꾸기
    if (end_idx - start_idx) % num != 0 :
        frame_end_idx += (num - end_idx%num) #hconcat 크기 맞추기 위함

    for i in range(start_idx,frame_end_idx):
        if i >= end_idx:
            img = np.zeros((112,112,3)) #empty imgs
        else: 
            img = dataset[i][0].permute(1,2,0).numpy() #(3,112,112) -> (112,112,3)
        offset = i - start_idx

        #가로로 먼저 쌓기
        if img_vertical is None: 
            img_vertical = img
            continue
        
        img_vertical = np.vstack([img_vertical,img]) 
        
        if (offset+1) % num == 0: #이제 세로로 쌓기
            if result is None :
                result = img_vertical
                img_vertical = None
                continue
            
            result = np.hstack([result, img_vertical])
            img_vertical = None

    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
