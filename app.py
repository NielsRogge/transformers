import streamlit as st
from PIL import Image
import os
import TDTSR
import pytesseract
from pytesseract import Output
import postprocess as pp
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')
st.title("Table Detection and Table Structure Recognition")

c1, c2, c3 = st.columns((1,1,1))


def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 

def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


for td_sample in os.listdir('D:/Jupyter/Multi-Type-TD-TSR/TD_samples/'):

    image = Image.open("D:/Jupyter/Multi-Type-TD-TSR/TD_samples/"+td_sample).convert("RGB")
    model, image, probas, bboxes_scaled = TDTSR.table_detector(image, THRESHOLD_PROBA=0.6)
    TDTSR.plot_results_detection(c1, model, image, probas, bboxes_scaled)
    cropped_img_list = TDTSR.plot_table_detection(c2, model, image, probas, bboxes_scaled)

    for table in cropped_img_list:
        # table : pil_img
        table = TDTSR.add_margin(table)
        model, image, probas, bboxes_scaled = TDTSR.table_struct_recog(table, THRESHOLD_PROBA=0.6)

        # The try, except block of code below plots table header row and simple rows
        try:
            header, row_header, rows, cols = TDTSR.plot_structure(c3, model, image, probas, bboxes_scaled, class_to_show=0)
            row_header, rows, cols = TDTSR.sort_table_features(header, row_header, rows, cols)
            # headers, rows, cols are ordered dictionaries with 5th element value of tuple being pil_img
            header, row_header, rows, cols = TDTSR.individual_table_features(table, header, row_header, rows, cols)
            TDTSR.plot_table_features(c1, header, row_header, rows, cols)
        except Exception as printableException: 
            st.write(td_sample, ' terminated with exception:', printableException)

        master_row = TDTSR.master_row_set(header, row_header, rows, cols)
        cells_img = TDTSR.object_to_cells(master_row, cols)
        total_rows = len(master_row)-1
        # Constructing column headers of dataframe:
        header_list = []
        header_idx = 0
        for tabular_feature, pil_img in cells_img.items():  
            
            if tabular_feature[:6] == 'header' and 'row' not in tabular_feature:
                header_text = ' '.join(pytesseract.image_to_data(pil_img, output_type=Output.DICT, config='preserve_interword_spaces')['text']).strip()
                if header_text == '':
                    header_list.append('empty_column_'+str(header_idx))
                    header_idx += 1
                else:
                    header_list.append(header_text)
                

        df = pd.DataFrame("", index=range(0, total_rows), columns=header_list)

        column_counter = 0

        for tabular_feature, pil_img in cells_img.items():  
            if tabular_feature[:9] == 'table row':
                column_counter = 0
                for cell_pil_img in pil_img:
                    cell_text = ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='preserve_interword_spaces')['text']).strip()
                    df.iat[int(tabular_feature.split('.')[-1]), column_counter] = cell_text
                    column_counter += 1

            elif tabular_feature[:16] == 'header table row':
                cell_text = ' '.join(pytesseract.image_to_data(pil_img, output_type=Output.DICT, config='preserve_interword_spaces')['text']).strip()
                df.iat[int(tabular_feature.split('.')[-1]), 0] = cell_text

            column_counter += 1

        c3.dataframe(df)




def OCRpreprocess():
    img = PIL_to_cv(pil_img)
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(img, bg, scale=255)
    # out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )[1] 
    pil_img = cv_to_PIL(out_gray)
    plt.imshow(pil_img)
    c1.pyplot()
    ############
    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    plt.imshow(pil_img)
    c1.pyplot()

    ############
    img = PIL_to_cv(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold the image using Otsu's thresholding method
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    pil_img = cv_to_PIL(thresh)
    plt.imshow(pil_img)
    c1.pyplot()
