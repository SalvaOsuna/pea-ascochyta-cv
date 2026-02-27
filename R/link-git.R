#1. Introduce Yourself to Git
library(usethis)
#2. Generate a New Personal Access Token (PAT)
use_git_config(user.name = "SalvaOsuna", user.email = "salvadorosunacaballero@gmail.com")
create_github_token()
#3. Store the Token in RStudio
library(gitcreds)
gitcreds_set()
#4. Sync Your Project
usethis::use_git()

#py check
 #install python and packages
library(reticulate)
use_python("C:/Program Files/Python3133/python.exe", required = TRUE)
repl_python()

library(reticulate)
py_install(c("opencv-python", "matplotlib", "numpy"))
reticulate::py_install("opencv-python")

reticulate::use_virtualenv("~/.virtualenvs/r-reticulate", required = TRUE)
#Switch from R to Python
repl_python()

print("Success!")
exit

#delete later
install.packages("FielDHub")
library(FielDHub)
run_app()
