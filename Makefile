CXX := nvcc
TARGET := vector_add
SRC := main.cu
CXXFLAGS := -O2

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

jupyter:
	uv run --with jupyter jupyter lab --allow-root --no-browser --NotebookApp.token=''
