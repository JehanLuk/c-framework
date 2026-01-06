CC = gcc
CFLAGS = -Wall -Wextra -g
SRC = $(wildcard src/*.c)
OUT = build/outDebug.exe

all:
	@if not exist build mkdir build
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

run: all
	$(OUT)

clean:
	del /Q build\*.exe 2>NUL
