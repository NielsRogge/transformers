import os
import cv2
from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import streamlit as st
from PIL import Image
import math


colors = ["red", "blue", "green", "yellow", "orange", "violet"]


def table_detector(image, THRESHOLD_PROBA):
  '''
  Table detection using DEtect-object TRansformer pre-trained on 1 million tables
  
  '''

  feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
  encoding = feature_extractor(image, return_tensors="pt")
  # encoding.keys()
  model = DetrForObjectDetection.from_pretrained("SalML\DETR-table-detection")
  # SalML\DETR-table-detection
  with torch.no_grad():
    outputs = model(**encoding)

  # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > THRESHOLD_PROBA

  # rescale bounding boxes
  target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
  postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
  bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

  return (model, image, probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
  '''
  Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
  '''

  feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
  encoding = feature_extractor(image, return_tensors="pt")

  model = DetrForObjectDetection.from_pretrained("SalML\DETR-table-structure-recognition")
  with torch.no_grad():
    outputs = model(**encoding)

  # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > THRESHOLD_PROBA

  # rescale bounding boxes
  target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
  postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
  bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

  return (model, image, probas[keep], bboxes_scaled)

def add_margin(pil_img, top=20, right=20, bottom=20, left=20, color=(255,255,255)):
  '''
  Image padding as part of TSR pre-processing to prevent missing table edges
  '''
  width, height = pil_img.size
  new_width = width + right + left
  new_height = height + top + bottom
  result = Image.new(pil_img.mode, (new_width, new_height), color)
  result.paste(pil_img, (left, top))
  return result

def plot_results_detection(c1, model, pil_img, prob, boxes, show_only_cropped=False):
  '''
  Plots the full pillow pdf-page image and adds a rectangle patch for table detection
  '''

  plt.figure(figsize=(32,20))
  plt.imshow(pil_img)
  ax = plt.gca()

  for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

      cl = p.argmax()
      xmin, ymin, xmax, ymax = xmin-3, ymin-3, xmax+3, ymax+3  
      ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=colors[cl.item()], linewidth=3))
      text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
      ax.text(xmin, ymin, text, fontsize=15,bbox=dict(facecolor='yellow', alpha=0.5))
  plt.axis('off')
  plt.show()
  c1.pyplot()


def plot_table_detection(c2, model, pil_img, prob, boxes):
  '''
  Plots only the cropped table(s) from the table detection 
  '''

  plt.figure(figsize=(32,20))
  ax = plt.gca()
  cropped_img_list = []

  for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

      xmin, ymin, xmax, ymax = xmin-3, ymin-3, xmax+3, ymax+3  
      cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
      cropped_img_list.append(cropped_img)

  for cropped_img in cropped_img_list:
    plt.imshow(cropped_img)

    plt.axis('off')
    plt.show()
    c2.pyplot()
  return cropped_img_list


def plot_structure(c3, model, pil_img, prob, boxes, class_to_show=0):
  '''
  To plot table pillow image and the TSR bounding boxes on the table
  '''
  plt.figure(figsize=(32,20))
  plt.imshow(pil_img)
  ax = plt.gca()
  rows = {}
  cols = {}
  header = {}
  row_header = {}
  idx = 0

  for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

      xmin, ymin, xmax, ymax = xmin-3, ymin-3, xmax+3, ymax+3  
      cl = p.argmax()
      class_text = model.config.id2label[cl.item()]
      text = f'{class_text}: {p[cl]:0.2f}'
      # st.write(class_text)
      if class_text != 'table':
        
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=colors[cl.item()], linewidth=3))
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

        if class_text == 'table column header':
          header['header'] = (xmin, ymin, xmax, ymax)
        if class_text == 'table row':
          rows['table row '+str(idx)] = (xmin, ymin, xmax, ymax)
        if class_text == 'table column':
          cols['table column '+str(idx)] = (xmin, ymin, xmax, ymax)
        if class_text == 'table projected row header':
          row_header['header table row'+str(idx)] = (xmin, ymin, xmax, ymax)

      idx += 1

  # plt.axis('off')
  plt.show()
  c3.pyplot()
  return header, row_header, rows, cols



def sort_table_features(header, row_header, rows, cols):
  # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
  y_header = header['header'][3] - 10
  rows_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1]) if ymin > y_header}
  cols_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}

  row_header_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(row_header.items(), key=lambda tup: tup[1][1])}

  new_row = {}
  idx = 0

  for k1, v1 in rows_.items():
    save_row = True
    row_xmin, row_ymin, row_xmax, row_ymax = v1
    for k2, v2 in row_header_.items():
      header_row_xmin, header_row_ymin, header_row_xmax, header_row_ymax = v2
      # table row and header table row are within 2 pixel range, skip saving the row
      if math.isclose(row_ymin, header_row_ymin, abs_tol=2):
        save_row = False
    if save_row:
      new_row['table row.'+str(idx)] = (row_xmin, row_ymin, row_xmax, row_ymax)
      idx += 1

  new_row_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(new_row.items(), key=lambda tup: tup[1][1])}

  return row_header_, new_row_, cols_

def individual_table_features(pil_img, header, row_header, rows, cols):

  for k, v in header.items():
    xmin, ymin, xmax, ymax = v
    cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
    header[k] = xmin, ymin, xmax, ymax, cropped_img

  for k, v in row_header.items():
    xmin, ymin, xmax, ymax = v
    cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
    row_header[k] = xmin, ymin, xmax, ymax, cropped_img

  for k, v in rows.items():
    xmin, ymin, xmax, ymax = v
    cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
    rows[k] = xmin, ymin, xmax, ymax, cropped_img


  for k, v in cols.items():
    xmin, ymin, xmax, ymax = v
    cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
    cols[k] = xmin, ymin, xmax, ymax, cropped_img

  return header, row_header, rows, cols

def plot_table_features(c2, header, row_header, rows, cols):

  for k, v in header.items():
    _, _, _, _, pil_img = v

    # c2.write('Table header:')
    # plt.imshow(pil_img)
    # plt.axis('off')
    # plt.show()
    # c2.pyplot()

  for k, v in row_header.items():
    _, _, _, _, pil_img = v

    # c2.write('Table row header:')
    # plt.imshow(pil_img)
    # plt.axis('off')
    # plt.show()
    # c2.pyplot()


  for k, v in rows.items():
    _, _, _, _, pil_img = v

    # c2.write('Rows:')
    # plt.imshow(pil_img)
    # plt.axis('off')
    # plt.show()
    # c2.pyplot()

  for k, v in cols.items():
    _, _, _, _, pil_img = v

  #   plt.imshow(pil_img)
  #   plt.axis('off')
  #   plt.show()
  #   c2.pyplot()

def master_row_set(header, row_header, rows, cols):
  master_row = {**header, **row_header, **rows}
  master_row_ = {table_feature : (xmin, ymin, xmax, ymax, img) for table_feature, (xmin, ymin, xmax, ymax, img) in sorted(master_row.items(), key=lambda tup: tup[1][1])}

  return master_row_


def object_to_cells(master_row, cols):
  '''
  Iterates to every row, be it header/simple row/header table row, cuts rows into cells and saves images in dictionary where length of dictionary = total rows
  '''
  cells_img = {}
  header_idx = 0
  row_idx = 0
  for k_row, v_row in master_row.items():

    if k_row[:16] == 'header table row':

      _, _, _, _, row_header_img = v_row
      cells_img[k_row+'.'+str(row_idx)] = row_header_img
      row_idx += 1

    elif k_row == 'header':

      _, ymin, _, ymax, header_img = v_row

      xa, ya, xb, yb = 0, 0, 0, ymax-ymin
      for k_col, v_col in cols.items():
        xmin_col, _, xmax_col, _, col_img = v_col
        xa = xmin_col-19
        xb = xmax_col-20

        header_img_cropped = header_img.crop((xa, ya, xb, yb))
        cells_img[k_row+'.'+str(header_idx)] = header_img_cropped
        header_idx += 1


    elif k_row[:9] == 'table row':

      xmin, ymin, xmax, ymax, row_img = v_row
      xa, ya, xb, yb = 0, 0, 0, ymax-ymin
      row_img_list = []
      for k_col, v_col in cols.items():
        xmin_col, _, xmax_col, _, col_img = v_col
        xa = xmin_col-19
        xb = xmax_col-20
        row_img_cropped = row_img.crop((xa, ya, xb, yb))
        row_img_list.append(row_img_cropped)
      cells_img[k_row+'.'+str(row_idx)] = row_img_list
      row_idx += 1
    
  return cells_img
