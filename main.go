package main

import (
	"fmt"
	"os/exec"
	"strings"
)

func recognizeImage(imagePath string) (string, error) {
	// 调用 Python 脚本进行图像识别
	cmd := exec.Command("python", "image_recognition.py", imagePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to execute command: %w", err)
	}

	// 解析输出
	result := strings.TrimSpace(string(output))
	return result, nil
}

func main() {
	imagePath := "./ma.jpeg"
	result, err := recognizeImage(imagePath)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Println("Recognized class:", result)
	}
}
