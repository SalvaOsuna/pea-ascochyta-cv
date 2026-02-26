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
library(reticulate)
repl_python()

print("Success!")
exit