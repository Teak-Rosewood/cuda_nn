CC = g++
CFLAGS = -Wall -Wextra -fopenmp

# Directory structure
SRCDIR = src
INCDIR = include
BUILDDIR = build
BINDIR = bin

# Files
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))
DEPS = $(OBJS:.o=.d)

# Targets
EXEC = $(BINDIR)/main

# Compiler flags
CPPFLAGS = -I$(INCDIR) -MMD -MP

.PHONY: all clean

all: $(EXEC)
	@echo "Success"

$(EXEC): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Include dependency files
-include $(DEPS)

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
