tmux start-server
tmux new-session -d -s do_console01 -n htop
tmux new-window -tdo_console01:1 -n sh
tmux new-window -tdo_console01:2 -n gpu
tmux new-window -tdo_console01:3 -n df


tmux send-keys -tdo_console01:0 'htop' C-m
tmux send-keys -tdo_console01:1 'cd bualabs' C-m
tmux send-keys -tdo_console01:2 'watch -n 1 nvidia-smi' C-m
tmux send-keys -tdo_console01:3 'df -h' C-m

tmux select-window -tdo_console01:0
tmux attach-session -d -tdo_console01
