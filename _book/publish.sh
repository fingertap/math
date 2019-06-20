#!/bin/bash

# Create branch for github pages
git branch -D gh-pages
git checkout --orphan gh-pages

# Enable mathjax
[[ -f "book.json" ]] || echo '{"plugins":["mathjax"]}' > book.json
[[ -d "node_modules" ]] || gitbook install .

# Replace $ with $$ for inline math blocks
find . -path ./node_modules -prune -o -name "*.md" -exec sed -i '' 's/\$/$$/g' {} +
find . -path ./node_modules -prune -o -name "*.md" -exec sed -i '' 's/\$\$\$\$/$$/g' {} +
gitbook build
rm -r node_modules
echo "_book" > .gitignore
git add .
git commit -m 'Math Delimiter'
# Remove files except _book
git rm --cache -r .
git clean -df
# Publish
cp -r _book/* .
git add .
git commit -m 'Publish'
git push -f origin gh-pages
# Cleaning
git checkout master
git branch -D gh-pages