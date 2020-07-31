ECHO AUTOSORT RUNNING, DO NOT CLOSE THIS WINDOW
call conda activate base
python "<updater script>"
python "<autosort script>" 1>"R:\Dannymarsh Sorting Emporium\logs\Autosort_%date:~-4,4%%date:~-7,2%%date:~-10,2%_.log" 2>"R:\Dannymarsh Sorting Emporium\logs\Autosort_errors_%date:~-4,4%%date:~-7,2%%date:~-10,2%_.log"
call conda deactivate
