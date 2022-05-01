import re

file_data = '0D0PrccF31F7F50FrccT0TrrrccS9A18B74S1A26B3S2A24B9S0A43B38W1W3W2E'
ctb_file_content =  re.split(r'[DPFFFFTTRRSABWWE]',file_data )

count_showdown = 0

def count_showd(file_data):
        
    for letter in file_data:
        
        if letter == 'S':
            global count_showdown
            count_showdown = count_showdown + 1
            
count_winners = 0

def count_winr(file_data):

    for letter in file_data:
        if letter == 'W':
            global count_winners
            count_winners = count_winners + 1




showdown = []
a = []
count_showd(file_data)
count_winr(file_data)

winners = []

for win in range(count_winners):
    winners.append(ctb_file_content[len(ctb_file_content) - (win + 2) ])

for show in range(0, count_showdown*3, 3):
    a = []
    for block in range(3):
        b = (ctb_file_content[len(ctb_file_content) - (len(winners) + show + block + 2)])
        a.append(b)
    a = list(reversed(a))
    showdown.append(a)

winners = list(reversed(winners))
showdown = list(reversed(showdown))


print(winners)
print(showdown)