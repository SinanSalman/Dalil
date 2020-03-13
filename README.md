# Dalil
**Dalil** is a tool for managing student paper exam electronic evidences; it merges and splits PDF scans of student exams, perform OCR of student IDs, cross-reference it with Blackboard grade center downloads to rename files with student IDs and names. The word Dalil is Arabic for Evidence.

Usage:

```shell
$ ./Dalil.py 
Loading config file...
Usage: Dalil.py [OPTIONS] COMMAND [ARGS]...

  Dalil is a tool for managing student paper exam electronic evidences; it
  merges and splits PDF  scans of student exams, perform OCR of student IDs,
  cross-reference it with Blackboard grade center  downloads to rename files
  with student IDs and names. The word Dalil is Arabic for Evidence.

Options:
  --help  Show this message and exit.

Commands:
  id       rename pdf file using read student ID
  merge    merge pdf files
  show_h   show header in pdf file by page
  show_id  show ID in pdf file by page
  split_c  split pdf file by page count
  split_h  split pdf file by header detection
  test     test pdf file IDs against confirmed file names and report results
  ```