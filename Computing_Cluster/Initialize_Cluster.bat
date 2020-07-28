ECHO AUTOSORT RUNNING, DO NOT CLOSE THIS WINDOW!!!
call conda activate base
python C:\Users\DiLorenzoTech\CCtransfer_final.py >"R:\Dannymarsh Sorting Emporium\CCtransfer_logs\CC_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log" 2>"R:\Dannymarsh Sorting Emporium\CCtransfer_logs\CC_error_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log"
call conda deactivate