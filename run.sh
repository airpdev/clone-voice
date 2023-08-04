if which python3.10 > /dev/null; then
  python3.10 main.py "$@"
else
  echo 'WARNING: python3.10 command was not found, attempting with python command, this could fail.'
  python main.py "$@"
fi
