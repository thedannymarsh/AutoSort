R:
ECHO AUTOSORT RUNNING, DO NOT CLOSE THIS WINDOW
call "C:\ProgramData\Anaconda3\Scripts\activate.bat" 
python "<updater script>"
python "<autosort script>" >"R:\Dannymarsh Sorting Emporium\logs\<name>_Autosort_%date:~-4,4%%date:~-7,2%%date:~-10,2%_.log" 2>"R:\Dannymarsh Sorting Emporium\logs\<name>_Autosort_error_%date:~-4,4%%date:~-7,2%%date:~-10,2%_.log"