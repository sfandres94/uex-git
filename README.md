<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]

# UEx-Git
This repository has been created for the lab session on Git and GitHub of the Multimedia Systems subject at the University of Extremadura (UEx).

## Table of contents
* [Basic commands](#basic-commands)
* [Gitignore](#gitignore)
* [Tagging](#tagging)
* [Working with branches](#working-with-branches)
* [Issues and pull requests](#issues-and-pull-requests)
* [License](#license)

## Basic commands
Add the new/modified/deleted files to the repo:
```
git add <filename>
```
Instead of `filename` you can use `.` or `--all` to add all the target files to the repository.

To delete a file from the repo:
```
git rm --cached <filename>
```
The `--cached` option removes the file from the staging area but remains intact inside the working directory.

To commit a change:
```
git commit -m "<comments>"
```

To push a commit from local to remote:
```
git push
```

Check the status of the repo:
```
git status
```

Reverts file changes:
```
git checkout -- <filename>
```

Move a file:
```
git mv <source> <destination>
```

Show commit logs:
```
git log
git log --oneline
```
Common flags: `-p <num>` to limit the output, `--stat`, `-S <function_name>`, `--since`, `--pretty`.

To discard all changes made after the specified commit (with hash):
```
git reset --hard <commit hash>
git push --force
```
The branch pointer is now moved to the specified commit, and all changes after that commit are discarded. Note that this operation cannot be undone, so be careful when using git reset.

Other options:
```
git restore, revert
```

## Gitignore
Difference between .gitignore rules with and without trailing slash like /dir and /dir/: [Link](https://stackoverflow.com/questions/17888695/difference-between-gitignore-rules-with-and-without-trailing-slash-like-dir-an)

## Tagging
Like most VCSs, Git can tag specific points in a repository’s history as being important. Typically, people use this functionality to mark release points (v1.0, v2.0, and so on).

Listing the existing tags in Git is straightforward. Just type the following with optional `-l` or `--list`:
```
git tag
```
```
git log
```

Creating tags:
```
git tag -a <number_of_version> -m "<tagging_message>"
```
where `number_of_version` can have the following format: v1.4, v1.8.5, etc.

Adding tags to existing commits:
```
git tag -a <number_of_version> <commit_hash> -m "<tagging_message>"
```
```
git push origin <number_of_version>
```

Credits to: [Link](https://git-scm.com/book/en/v2/Git-Basics-Tagging)

## Working with branches
* [Information on branching and merging](https://nvie.com/posts/a-successful-git-branching-model/)

## Issues and pull requests
* [Reference code in issues](https://geeks.ms/jorge/2017/08/26/marcar-un-codigo-en-github-para-hacer-referencia-comentar-o-compartir/)
* [Information on pull requests](https://www.freecodecamp.org/espanol/news/como-hacer-tu-primer-pull-request-en-github/)

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/sfandres
