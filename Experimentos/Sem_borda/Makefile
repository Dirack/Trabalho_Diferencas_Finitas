# Makefile
# 
# Objetivo: Compilar o programa semborda.x com API do Madagascar.
# 
# Site: https://dirack.github.io
# 
# Versão 1.0
# 
# Programador: Rodolfo A C Neves (Dirack) 20/08/2021
# 
# Email: rodolfo_profissional@hotmail.com
# 
# Licença: GPL-3.0 <https://www.gnu.org/licenses/gpl-3.0.txt>.


all:	semborda.x

semborda.x:	Msemborda.c
	gcc -g $< -I$$RSFSRC/include -I.. -L$$RSFSRC/lib -o $@ -lrsf -lm

clean:
	rm *.x

help:
	@echo "Para compilar o exemplo use 'make'"
