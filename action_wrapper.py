def mario_action_interpret(action):
    switcher = {
        0: 'nop',
        1: 'right',
        2: 'right, A',
        3: 'right, B',
        4: 'right, A, B',
        5: 'A',
        6: 'B'
    }
    return switcher.get(action)
