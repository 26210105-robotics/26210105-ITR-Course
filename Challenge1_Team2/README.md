**Challenge 1**  
   
python3 -c "import json; print(json.load(open('omx_calibration.json'))\['offset_mm'\])"  
   
python3 calib_check.py omx_calibration.json --check  
