package main

import (
	"fmt"
	"sync"
	"time"
)

func anotherFuntion(val int) {
	time.Sleep(time.Second * 1)
	fmt.Println("fucntions completed: ", val)
}

func wrapperAnotherFucntion(val int, wg *sync.WaitGroup) {

	defer wg.Done()
	anotherFuntion(val)
}

func secondFunction(fn func(int, *sync.WaitGroup), val2 int, val3 int) {

	var wg sync.WaitGroup
	for i := 0; i < val3; i++ {
		wg.Add(1)
		go fn(i, &wg)
	}

	wg.Wait()

	fmt.Println("Second fucntion compeletd")
}

func main() {
	secondFunction(wrapperAnotherFucntion, 2, 10)
}
