#!/bin/sh

while getopts 'hf' opt
do
	case "$opt" in
		h)
			echo "usage: $0 [-hf]"
			echo ''
			echo 'Push documentation to Github Pages.'
			echo ''
			echo 'This will push the directory `docs/_build/html` to the branch'
			echo 'named `gh-pages` on the remote named `origin`.'
			echo ''
			echo 'This script should be called from the root of the repository.'
			echo ''
			echo 'Options'
			echo '    -h    Print this help message then exit.'
			echo '    -f    Delete the remote branch before pushing.'
			echo ''
			exit 0
			;;
		f)
			git push --delete origin gh-pages
			;;
	esac
done

git subtree push --prefix docs/_build/html origin gh-pages
