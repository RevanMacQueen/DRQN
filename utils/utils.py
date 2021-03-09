def to_command(dic):
    command = 'python3 main.py'
    for key, value in dic.items():
        if key == 'only_reward' or key =='save_recording' :
            command += ' --{}'.format(key)
        else:
            command += ' --{} {}'.format(key, value)

    return command + '\n'