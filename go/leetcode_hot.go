package main

import (
	"fmt"
)

func rotate(nums []int, k int) []int {
	k = k % len(nums)
	reverse := func(start, end int) {
		for start < end {
			nums[start], nums[end] = nums[end], nums[start]
			start++
			end--
		}
	}
	// slices.Reverse(nums)
	reverse(0, len(nums)-1)
	reverse(0, k-1)
	reverse(k, len(nums)-1)
	return nums
}

func case189() {
	nums := []int{1, 2, 3, 4, 5, 6, 7}
	fmt.Println(rotate(nums, 3))
}

func main() {
	case189()
}
