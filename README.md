# Table transformer to CSV

NOTE: This fork also contains code for launching your streamlit application to convert PDF-page-images that contain tables, to pandas dataframe that can be visualized and saved.

> Credits: Creators of table-transformer (MSR) https://github.com/microsoft/table-transformer
and https://github.com/NielsRogge/ who Integrated table-transformer to HuggingFace.

> My contribution: Added code to extract bounding boxes from the TSR model, use pytesseract and project to pandas dataframe.

The OCR is done by pytesseract (on Windows 11), please make necessary changes to the image paths and model checkpoint paths.
The model checkpoint can be found on HuggingFace: https://huggingface.co/nielsr/detr-table-detection and https://huggingface.co/nielsr/detr-table-structure-recognition


## Dashboard overview:
![image](https://user-images.githubusercontent.com/31631107/189661386-2a173cf4-9d0a-4c3b-9877-4f6439ed119d.png)
## Specific example:
### Table detection
![image](https://user-images.githubusercontent.com/31631107/189661619-1bac28bd-1d56-4989-81e6-726e93409194.png)
### Table Structure Recognition
![image](https://user-images.githubusercontent.com/31631107/189661745-32bb808d-e1a0-4549-bf41-2494d010ace2.png)
### Pandas Dataframe (can be exported to CSV)
![image](https://user-images.githubusercontent.com/31631107/189662133-272ce876-8efa-44de-bb43-8f900b3ee0bc.png)

