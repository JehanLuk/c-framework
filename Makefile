CC = gcc

CFLAGS = -Wall -Wextra -g -Iinclude
RELEASE_FLAGS = -Wall -Wextra -O2 -Iinclude

SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,obj/%.o,$(SRC))

DEBUG_OUT = build/outDebug.exe
RELEASE_OUT = bin/mlinc.exe

.PHONY: all debug release clean run

all: debug

debug: $(DEBUG_OUT)

release: $(RELEASE_OUT)

# Create the necessary folders
obj:
	mkdir -p obj

build:
	mkdir -p build

bin:
	mkdir -p bin

# Compile .c archives
obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

# Debug
$(DEBUG_OUT): $(OBJ) | build
	$(CC) $(OBJ) -o $@ -lm

# Release
$(RELEASE_OUT): $(OBJ) | bin
	$(CC) $(OBJ) $(RELEASE_FLAGS) -o $@ -lm

# Clean
clean:
	rm -f $(OBJ)
	rm -f $(DEBUG_OUT)
	rm -f $(RELEASE_OUT)

# Execute the debug version
run: debug
	./$(DEBUG_OUT)