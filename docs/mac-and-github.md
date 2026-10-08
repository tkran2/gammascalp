# Mac setup and GitHub publishing

The archive contains Python source, the results of the delivered run, and the
memo. Nothing needs compiling.

## Download and run

Download `gammascalp.zip` into your Downloads folder. In Terminal:

```bash
mkdir -p ~/Projects
unzip ~/Downloads/gammascalp.zip -d ~/Projects
cd ~/Projects/gammascalp
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
python -m gammascalp hedge --rebalances 63 --paths 50000
```

Python 3.10 or newer is required. If `python3` is missing and you have
Homebrew, run `brew install python`. Each new Terminal window needs
`source .venv/bin/activate` again before running the commands.

## Publish to GitHub from Terminal

Install the GitHub CLI if `gh --version` does not work:

```bash
brew install gh
```

If Homebrew is not installed, use the official instructions at
https://brew.sh/, then install `gh`.

Authenticate using the interactive terminal flow:

```bash
gh auth login --scopes workflow
gh auth setup-git
```

Choose GitHub.com and HTTPS. The flow offers browser sign-in or token sign-in.
The extra scope permits uploading the included GitHub Actions workflow.

Create and publish the repository:

```bash
cd ~/Projects/gammascalp
git init -b main
git add .
git -c user.name='Tom Krane' -c user.email='tkran2@users.noreply.github.com' commit -m 'Build option hedging simulator with theory checks and research memo'
gh repo create gammascalp --public --source=. --remote=origin --push
gh repo view --json url --jq .url
```

The command creates a public repository under the account selected by
`gh auth login`. If `gammascalp` already exists, choose a different unused
name in the `gh repo create` command. It will not overwrite an existing repo.

## Save later changes

```bash
python -m unittest discover -s tests
git add .
git commit -m 'Describe the change you made'
git push
```

Only the initial commit above supplies a name and email for that one command.
If your Mac's Git identity is not configured, configure it before later commits:

```bash
git config user.name 'Tom Krane'
git config user.email 'tkran2@users.noreply.github.com'
```

These settings apply to this repository. The noreply email avoids publishing
your school email in Git commit metadata.
