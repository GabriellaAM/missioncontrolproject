with open('soros_system/portfolio/backtest.py', 'r') as f:
    content = f.read()

content = content.replace('        else:', '            else:')
content = content.replace('        running_max', '            running_max')
content = content.replace('        annualized_return', '            annualized_return')

with open('soros_system/portfolio/backtest.py', 'w') as f:
    f.write(content)
print('Fix applied')
