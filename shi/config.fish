if status is-interactive
    # Commands to run in interactive sessions can go here
end

fish_add_path /usr/local/bin/
fish_add_path ~/.local/bin/
fish_add_path ~/.config/emacs/bin/
set -U fish_prompt_pwd_dir_length 0

function prompt_hostname
    echo -n "hostname-not-set"
end

# custom aliases
alias config "emacs -nw ~/.config/fish/config.fish"
alias e "emacs -nw"
alias emacs "emacs -nw"
alias gits "git status"
alias s "source ~/.config/fish/config.fish"
alias xx "cd ~/miles/"

# key bindings
bind right forward-single-char

