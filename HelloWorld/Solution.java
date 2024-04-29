import java.util.*;
import java.lang.reflect.Array;
import java.util.regex.MatchResult;

class Solution {

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public class Node {
        int val;
        Node next;
        Node random;

        Node() {
        }

        Node(int val) {
            this.val = val;
        }

        Node(int val, Node next) {
            this.val = val;
            this.next = next;
        }
    }

    public void dfs(char[][] grid, int n, int m, int i, int j) {
        grid[i][j] = '#';

        int[][] dxy = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
        for (int k = 0; k < 4; k++) {
            int dx = i + dxy[k][0];
            int dy = j + dxy[k][1];
            if (dx >= 0 && dx < n && dy >= 0 && dy < m && grid[dx][dy] == '1') {
                dfs(grid, n, m, dx, dy);
            }
        }
        return;
    }

    public int findMaxLength(int[] nums) {
        int n = nums.length;
        HashMap<Integer, Integer> prefixHashMap = new HashMap<>();
        int maxLength = 0;
        int currSum = 0;
        prefixHashMap.put(0, 0);
        for (int i = 0; i < n; i++) {
            currSum += nums[i] == 1 ? 1 : -1;
            if (prefixHashMap.containsKey(currSum)) {
                maxLength = Math.max(maxLength, i - prefixHashMap.get(currSum));
            } else {
                prefixHashMap.put(currSum, i);
            }
        }
        return maxLength;
    }

    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        int countOf0 = 0;
        int product = 1;
        for (int num : nums) {
            if (num == 0) {
                countOf0++;
            } else {
                product *= num;
            }
        }
        if (countOf0 > 1) {
            return result;
        }
        for (int i = 0; i < n; i++) {
            if (countOf0 == 1) {
                if (nums[i] == 0) {
                    result[i] = product;
                }
            } else {
                result[i] = product / nums[i];
            }
        }
        return result;
    }

    public int numSubarraysWithSum(int[] nums, int goal) {
        int n = nums.length;
        int count = 0;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = i; j < n; j++) {
                sum += nums[j];
                if (sum == goal) {
                    count++;
                }
            }
        }
        return count;
    }

    public int pivotInteger(int n) {
        int low = 1;
        int high = n;
        int totalSum = (n * (n + 1)) / 2;
        int rootVal = (int) Math.sqrt(totalSum);
        if (rootVal * rootVal == totalSum) {
            return rootVal;
        }
        return -1;
    }

    public void removeNode(ListNode starting, ListNode ending, HashMap<Integer, ListNode> sequenceSum, int prevSum) {
        int currSum = prevSum;
        while (starting != ending) {
            currSum += starting.val;
            sequenceSum.remove(currSum);
            starting = starting.next;
        }
    }

    public ListNode removeZeroSumSublists(ListNode head) {
        HashMap<Integer, ListNode> sequenceSum = new HashMap<>();

        sequenceSum.put(head.val, head);
        ListNode temp = head.next;
        int currSum = head.val;

        while (temp != null) {
            currSum += temp.val;
            if (currSum == 0) {
                if (temp.next != null) {
                    head.val = temp.next.val;
                    head.next = temp.next.next;
                    removeNode(head, temp, sequenceSum, 0);
                } else {
                    return null;
                }
            } else if (sequenceSum.containsKey(currSum)) {
                ListNode startingNode = sequenceSum.get(currSum);
                removeNode(startingNode, temp, sequenceSum, currSum);
                startingNode.next = temp.next;
            } else {
                sequenceSum.put(currSum, temp);
            }
            temp = temp.next;
        }
        return head;
    }

    public boolean isPresent(int[] nums, int target) {
        int n = nums.length;
        int low = 0, high = n - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) {
                return true;
            } else if (nums[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return false;
    }

    static int equalPartition(int N, int arr[]) {
        int sum = 0;
        for (int num : arr) {
            sum += num;
        }

        if (sum % 2 == 1) {
            return 0;
        }
        sum = sum / 2;

        boolean[][] subSetSum = new boolean[N + 1][sum + 1];
        for (int i = 0; i < N + 1; i++) {
            subSetSum[i][0] = true;
        }
        for (int i = 1; i < sum + 1; i++) {
            subSetSum[0][i] = false;
        }

        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < sum + 1; j++) {
                if (arr[i - 1] <= j) {
                    subSetSum[i][j] = (subSetSum[i - 1][j - arr[i - 1]] || subSetSum[i - 1][j]);
                } else {
                    subSetSum[i][j] = subSetSum[i - 1][j];
                }
            }
        }
        return subSetSum[N][sum] == true ? 1 : 0;

    }

    static Boolean isSubsetSum(int N, int arr[], int sum) {
        // code here

        boolean[][] subSetSum = new boolean[N + 1][sum + 1];

        for (int i = 0; i < N + 1; i++) {
            subSetSum[i][0] = true;
        }
        for (int i = 1; i < sum + 1; i++) {
            subSetSum[0][i] = false;
        }

        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < sum + 1; j++) {
                if (arr[i - 1] <= j) {
                    subSetSum[i][j] = (subSetSum[i - 1][j - arr[i - 1]] || subSetSum[i - 1][j]);
                } else {
                    subSetSum[i][j] = subSetSum[i - 1][j];
                }
            }
        }
        return subSetSum[N][sum];
    }

    public int perfectSum(int arr[], int n, int sum) {

        int mod = 1_000_000_007;
        int[][] prefectSumCount = new int[n + 1][sum + 1];
        for (int i = 0; i < n + 1; i++) {
            prefectSumCount[i][0] = 1;
        }

        for (int i = 1; i < sum + 1; i++) {
            prefectSumCount[0][i] = 0;
        }

        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < sum + 1; j++) {
                if (arr[i - 1] <= j) {
                    prefectSumCount[i][j] = (prefectSumCount[i - 1][j - arr[i - 1]] + prefectSumCount[i - 1][j]) % mod;
                } else {
                    prefectSumCount[i][j] = prefectSumCount[i - 1][j];
                }
            }
        }
        return prefectSumCount[n][sum];
    }

    public boolean[] subSetSum(int arr[], int n, int sum) {
        boolean subSetSum[][] = new boolean[n + 1][sum + 1];

        for (int i = 0; i < n + 1; i++) {
            subSetSum[i][0] = true;
        }

        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < sum + 1; j++) {
                if (arr[i - 1] <= j) {
                    subSetSum[i][j] = subSetSum[i - 1][j - arr[i - 1]] || subSetSum[i - 1][j];
                } else {
                    subSetSum[i][j] = subSetSum[i - 1][j];
                }
            }
        }
        return subSetSum[n];
    }

    public int minDifference(int arr[], int n) {
        int high = 0;
        for (int num : arr) {
            high += num;
        }

        boolean[] subSetSum = subSetSum(arr, n, high / 2);

        int minDiff = Integer.MAX_VALUE;
        for (int i = 0; i <= high / 2; i++) {
            if (subSetSum[i] == true) {
                minDiff = Math.min(minDiff, high - 2 * i);
            }
        }
        return minDiff;
    }

    public int countOfSubSetSum(int[] arr, int n, int sum) {
        int[][] dp = new int[n + 1][sum + 1];
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = 1;
        }
        for (int j = 1; j < sum + 1; j++) {
            dp[0][j] = 0;
        }

        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < sum + 1; j++) {
                if (arr[i - 1] <= j) {
                    dp[i][j] = dp[i - 1][j - arr[i - 1]] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][sum];
    }

    public int countPartitions(int n, int d, int arr[]) {

        int range = 0;
        for (int num : arr) {
            range += num;
        }

        int sum = (range - d) / 2;
        return countOfSubSetSum(arr, n, sum);
    }

    public int testingFucntion(int n) {
        return n >> 1;
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {
        int n = intervals.length;
        ArrayList<int[]> newList = new ArrayList<>();
        int i = 0;
        while (i < n && intervals[i][1] < newInterval[0]) {
            newList.add(intervals[i]);
            i++;
        }

        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(intervals[i][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[i][1], newInterval[1]);
            i++;
        }
        newList.add(newInterval);
        while (i < n) {
            newList.add(intervals[i]);
            i++;
        }
        return newList.toArray(new int[newList.size()][]);
    }

    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> Integer.compare(a[1], b[1]));
        int n = points.length;
        if (n < 1) {
            return 0;
        }
        int arrows = 0, currValid = points[0][1];
        for (int i = 1; i < n; i++) {
            if (points[i][0] > currValid) {
                arrows++;
                currValid = points[i][1];
            }
        }
        arrows++;
        return arrows;
    }

    public int leastInterval(char[] tasks, int n) {
        int[] taskScheduler = new int[26];
        for (char task : tasks) {
            taskScheduler[task - 'A']++;
        }
        // first are finding total idle spot after filling greatest occuring element and
        // filling rest of the elements in the idle spots

        Arrays.sort(taskScheduler);
        // maxFreq = -1 as last node don't have to wait
        int maxFreq = taskScheduler[25] - 1;
        int idleSpots = maxFreq * n;

        for (int i = 24; i >= 0 && taskScheduler[i] > 0; i--) {
            idleSpots -= Math.min(maxFreq, taskScheduler[i]);
        }
        return idleSpots > 0 ? tasks.length + idleSpots : tasks.length;
    }

    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        ListNode pergeNode = list1;
        for (int i = 0; i < b + 1 && pergeNode != null; i++) {
            pergeNode = pergeNode.next;
        }
        ListNode startNode = list1;
        for (int i = 0; i < a - 1; i++) {
            startNode = startNode.next;
        }
        startNode.next = list2;
        while (startNode.next != null) {
            startNode = startNode.next;
        }
        startNode.next = pergeNode;
        return list1;
    }

    public int minPairSum(int[] nums) {
        Arrays.sort(nums);
        int minValue = Integer.MAX_VALUE;
        int start = 0;
        int end = nums.length - 1;
        while (start < end) {
            minValue = Math.min(minValue, nums[start] + nums[end]);
            start++;
            end--;
        }
        return minValue;
    }

    public ListNode reverse(ListNode head) {
        ListNode temp = null;
        while (head != null) {
            ListNode curr = head.next;
            head.next = temp;
            temp = head;
            head = curr;
        }
        return temp;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        slow = reverse(slow.next);
        while (slow != null) {
            if (head.val != slow.val) {
                return false;
            }
            head = head.next;
            slow = slow.next;
        }
        return true;
    }

    public ListNode reverseList(ListNode head) {
        ListNode temp = null;
        while (head != null) {
            ListNode curr = head.next;
            head.next = temp;
            temp = head;
            head = curr;
        }
        return temp;
    }

    public void reorderList(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = reverse(slow.next);
        slow.next = null;
        while (fast != null) {
            ListNode curr = head.next;
            head.next = fast;
            fast = fast.next;
            head.next.next = curr;
            head = curr;
        }
    }

    public int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];

        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        fast = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> duplicates = new ArrayList<>();
        int n = nums.length;
        for (int i = 1; i < n + 1; i++) {
            if (nums[Math.abs(nums[i - 1])] < 0) {
                duplicates.add(Math.abs(nums[i - 1]));
            } else {
                nums[Math.abs(nums[i - 1])] *= -1;
            }
        }
        return duplicates;
    }

    public static int frogJump(int n, int heights[]) {
        if (n == 1) {
            return 0;
        }
        // int minEnergy = Integer.MAX_VALUE;
        int[] dp = new int[n];
        dp[0] = 0;
        dp[1] = Math.abs(heights[0] - heights[1]);
        int prev2 = 0;
        int prev = Math.abs(heights[0] - heights[1]);
        for (int i = 2; i < n; i++) {
            int left = prev + Math.abs(heights[i] - heights[i - 1]);
            int right = prev2 + Math.abs(heights[i] - heights[i - 2]);
            int curri = Math.min(left, right);
            prev2 = prev;
            prev = curri;
        }
        return prev;
    }

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[i] - 1 != i && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[i];
                nums[i] = nums[temp - 1];
                nums[temp - 1] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] - 1 != i) {
                return i + 1;
            }
        }
        return n + 1;
    }

    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int start = 0, end = 0, arrayProduct = 1, countOfSumArray = 0;
        while (end < nums.length) {
            arrayProduct *= nums[end];
            end++;
            while (start < end && arrayProduct >= k) {
                arrayProduct /= nums[start];
                start++;
            }
            if (arrayProduct < k) {
                countOfSumArray += end - start;
            }
        }
        return countOfSumArray;
    }

    public int climbStairs(int n) {
        int prev = 1;
        int prev2 = 0;
        for (int i = 1; i <= n; i++) {
            int curri = prev + prev2;
            prev2 = prev;
            prev = curri;
        }
        return prev;
    }

    public int maxSubarrayLength(int[] nums, int k) {
        int n = nums.length;
        HashMap<Integer, Integer> sequenceLenth = new HashMap<>();
        int left = 0, right = 0, maxLength = 0;
        while (left < n && right < n) {
            sequenceLenth.put(nums[right], sequenceLenth.getOrDefault(sequenceLenth, 0) + 1);
            while (sequenceLenth.get(nums[right]) > k) {
                sequenceLenth.put(nums[left], sequenceLenth.getOrDefault(nums[left], 0) - 1);
                left++;
            }
            right++;
            maxLength = Math.max(maxLength, right - left);
        }
        return maxLength;
    }

    public int findMaxMoney(int idx, int[] nums, int[] dp) {
        if (idx == 0)
            return nums[idx];
        if (idx < 0)
            return 0;

        if (dp[idx] != -1)
            return dp[idx];
        int pick = nums[idx] + findMaxMoney(idx - 2, nums, dp);
        int notPick = findMaxMoney(idx - 1, nums, dp);
        return dp[idx] = Math.max(pick, notPick);
    }

    public int rob(int[] nums, int start, int end) {
        int prev = nums[start];
        int prev2 = 0;
        for (int idx = start + 1; idx < end; idx++) {
            int pick = nums[idx];
            if (idx > 1)
                pick += prev2;
            int notPick = prev;
            int curri = Math.max(pick, notPick);
            prev2 = prev;
            prev = curri;
        }
        return prev;
    }

    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }
        int selectFirstHouse = rob(nums, 0, n - 1);
        int selectLastHouse = rob(nums, 1, n);
        return Math.max(selectFirstHouse, selectLastHouse);
    }

    public static int ninjaTraining(int day, int last, int[][] points, int[][] dp) {
        if (dp[day][last] != -1) {
            return dp[day][last];
        }
        if (day == 0) {
            int maxi = 0;
            for (int i = 0; i < 3; i++) {
                if (i != last) {
                    maxi = Math.max(maxi, points[0][i]);
                }
            }
            return maxi;
        }
        int maaxi = 0;
        for (int i = 0; i < 3; i++) {
            if (i != last) {
                int activity = points[day][i] + ninjaTraining(day - 1, i, points, dp);
                maaxi = Math.max(maaxi, activity);
            }
        }
        return dp[day][last] = maaxi;
    }

    public static int ninjaTraining(int n, int points[][]) {
        int[] prev = new int[4];

        prev[0] = Math.max(points[0][1], points[0][2]);
        prev[1] = Math.max(points[0][0], points[0][2]);
        prev[2] = Math.max(points[0][0], points[0][1]);
        prev[2] = Math.max(points[0][0], Math.max(points[0][1], points[0][2]));

        for (int day = 1; day < n; day++) {
            int[] temp = new int[4];
            for (int last = 0; last < 4; last++) {
                temp[last] = 0;
                for (int task = 0; task < 3; task++) {
                    if (task != last) {
                        int activity = points[day][task] + prev[task];
                        temp[last] = Math.max(temp[last], activity);
                    }
                }
            }
            prev = temp;
        }
        return prev[3];
    }

    public long countSubarrays(int[] nums, int k) {
        int size = nums.length, left = 0, right = 0;
        int maxElement = Integer.MIN_VALUE;
        for (int row : nums) {
            maxElement = Math.max(maxElement, row);
        }
        int subArrayCount = 0;
        long ans = 0;
        while (right < size) {
            if (nums[right++] == maxElement) {
                subArrayCount++;
            }
            while (subArrayCount == k) {
                if (nums[left++] == maxElement) {
                    subArrayCount--;
                }
            }
            ans += left;
        }
        return ans;
    }

    public int subarraysWithKDistinct(int[] nums, int k) {
        return subarraysWithAtmostKDistinct(nums, k) - subarraysWithAtmostKDistinct(nums, k - 1);
    }

    public int subarraysWithAtmostKDistinct(int[] nums, int k) {
        HashMap<Integer, Integer> freqMap = new HashMap<>();
        int left = 0, right = 0, size = nums.length, currDistinct = 0, ans = 0;
        while (right < size) {
            int currNumCount = freqMap.getOrDefault(nums[right], 0);
            if (currNumCount == 0) {
                currDistinct++;
            }
            freqMap.put(nums[right], currNumCount + 1);
            while (currDistinct > k) {
                freqMap.put(nums[left], freqMap.get(nums[left]) - 1);
                if (freqMap.get(nums[left]) == 0) {
                    currDistinct--;
                }
                left++;
            }
            ans += right - left + 1;
        }
        return ans;
    }

    public long countSubarrays(int[] nums, int minK, int maxK) {
        int minIdx = -1, maxIdx = -1, badIdx = -1;
        long count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (!(minK <= nums[i] && nums[i] <= maxK)) {
                badIdx = i;
            }

            if (minK == nums[i])
                minIdx = i;

            if (maxK == nums[i])
                maxIdx = i;

            count += Math.max(0, Math.min(minIdx, maxIdx) - badIdx);
        }
        return count;
    }

    public int uniquePathsCount(int i, int j, int[][] dp) {
        if (i == 0 || j == 0) {
            return 1;
        }
        if (i < 0 || j < 0) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }

        int up = uniquePathsCount(i - 1, j, dp);
        int left = uniquePathsCount(i, j - 1, dp);
        return dp[i][j] = up + left;
    }

    public int uniquePathsTabulation(int m, int n) {
        int dp[][] = new int[m][n];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }
        dp[0][0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[0][0] = 1;
                    continue;
                }
                int up = 0, left = 0;
                if (i - 1 >= 0)
                    up = dp[i - 1][j];
                if (j - 1 >= 0)
                    left = dp[i][j - 1];
                dp[i][j] = up + left;
            }
        }
        return dp[m - 1][n - 1];
    }

    public int uniquePaths(int m, int n) {
        int prev[] = new int[n];
        for (int i = 0; i < m; i++) {
            int[] temp = new int[n];
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    temp[0] = 1;
                    continue;
                }
                int up = 0, left = 0;
                if (i - 1 >= 0)
                    up = prev[j];
                if (j - 1 >= 0)
                    left = temp[j - 1];
                temp[j] = up + left;
            }
            prev = temp;
        }
        return prev[n - 1];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int prev[] = new int[n];
        for (int i = 0; i < m; i++) {
            int[] curr = new int[n];
            for (int j = 0; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    curr[j] = 0;
                } else if (i == 0 && j == 0) {
                    curr[j] = 1;
                } else {
                    int up = 0, left = 0;
                    if ((i - 1) >= 0)
                        up = prev[j];
                    if ((j - 1) >= 0)
                        left = curr[j - 1];
                    curr[j] = up + left;
                }
            }
            prev = curr;
        }
        return prev[n - 1];
    }

    public int minPathSumRecursion(int i, int j, int[][] grid, int[][] dp) {
        if (i == 0 && j == 0) {
            return grid[i][j];
        }
        if (i < 0 || j < 0) {
            return 1_000_000_000;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        int up = grid[i][j] + minPathSumRecursion(i - 1, j, grid, dp);
        int left = grid[i][j] + minPathSumRecursion(i, j - 1, grid, dp);
        return dp[i][j] = Math.min(up, left);
    }

    public int minPathSumMemorization(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }
        return minPathSumRecursion(m - 1, n - 1, grid, dp);
    }

    public int minPathSumTabulation(int[][] grid) {
        int billion = 1_000_000_000;
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[i][j];
                } else {
                    int up = grid[i][j], left = grid[i][j];
                    if ((i - 1) >= 0) {
                        up += dp[i - 1][j];
                    } else {
                        up += billion;
                    }
                    if ((j - 1) >= 0) {
                        left += dp[i][j - 1];
                    } else {
                        left += billion;
                    }
                    dp[i][j] = Math.min(up, left);
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public int minPathSumSpaceOptimized(int[][] grid) {
        int billion = 1_000_000_000;
        int m = grid.length;
        int n = grid[0].length;
        int[] prev = new int[n];
        for (int i = 0; i < m; i++) {
            int curr[] = new int[n];
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    curr[j] = grid[i][j];
                } else {
                    int up = grid[i][j], left = grid[i][j];
                    if ((i - 1) >= 0) {
                        up += prev[j];
                    } else {
                        up += billion;
                    }
                    if ((j - 1) >= 0) {
                        left += curr[j - 1];
                    } else {
                        left += billion;
                    }
                    curr[j] = Math.min(up, left);
                }
            }
            prev = curr;
        }
        return prev[n - 1];
    }

    public int minimumTotalRecursive(int i, int j, List<List<Integer>> triangle, int n, int[][] dp) {
        if (i == n - 1) {
            return triangle.get(n - 1).get(j);
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        int dl = triangle.get(i).get(j) + minimumTotalRecursive(i + 1, j, triangle, n, dp);
        int dg = triangle.get(i).get(j) + minimumTotalRecursive(i + 1, j + 1, triangle, n, dp);
        return dp[i][j] = Integer.min(dl, dg);
    }

    public int minimumTotalMemo(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[][] dp = new int[n][n];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }
        return minimumTotalRecursive(0, 0, triangle, n, dp);
    }

    public int minimumTotalTabulation(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[][] dp = new int[n][n];
        for (int j = 0; j < n; j++) {
            dp[n - 1][j] = triangle.get(n - 1).get(j);
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i; j >= 0; j--) {
                int dl = triangle.get(i).get(j) + dp[i + 1][j];
                int dg = triangle.get(i).get(j) + dp[i + 1][j + 1];
                dp[i][j] = Math.min(dl, dg);
            }
        }
        return dp[0][0];
    }

    public int minimumTotalSpaceOptimized(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] front = new int[n];
        int[] curr = new int[n];
        for (int j = 0; j < n; j++) {
            front[j] = triangle.get(n - 1).get(j);
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i; j >= 0; j--) {
                int dl = triangle.get(i).get(j) + front[j];
                int dg = triangle.get(i).get(j) + front[j + 1];
                curr[j] = Math.min(dl, dg);
            }
            front = curr;
        }
        return front[0];
    }

    public int lengthOfLastWord(String s) {
        int length = s.length();
        int ans = 0;
        for (int i = length - 1; i >= 0; i--) {
            char currCharacter = s.charAt(i);
            if (!Character.isWhitespace(currCharacter)) {
                ans += 1;
            } else if (ans > 0) {
                break;
            }
        }
        return ans;
    }

    public int minFallingPathSumRecursion(int i, int j, int[][] matrix, int[][] dp) {
        if (j < 0 || j >= matrix[0].length) {
            return 1_000_000_000;
        }
        if (i == 0) {
            return matrix[0][j];
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        int up = matrix[i][j] + minFallingPathSumRecursion(i - 1, j, matrix, dp);
        int dl = matrix[i][j] + minFallingPathSumRecursion(i - 1, j - 1, matrix, dp);
        int dr = matrix[i][j] + minFallingPathSumRecursion(i - 1, j + 1, matrix, dp);
        return dp[i][j] = Math.min(up, Math.min(dl, dr));
    }

    public int minFallingPathSumMemo(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] dp = new int[n][m];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }
        int mini = Integer.MAX_VALUE;
        for (int j = 0; j < m; j++) {
            int ans = minFallingPathSumRecursion(n - 1, j, matrix, dp);
            mini = Math.min(mini, ans);
        }
        return mini;
    }

    public int minFallingPathSumTabulation(int[][] matrix) {
        int billion = 1_000_000_000;
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] dp = new int[n][m];
        for (int j = 0; j < m; j++) {
            dp[0][j] = matrix[0][j];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int up = matrix[i][j] + dp[i - 1][j];
                int dl = matrix[i][j], dr = matrix[i][j];
                if (j - 1 >= 0)
                    dl += dp[i - 1][j - 1];
                else
                    dl += billion;

                if (j + 1 < m)
                    dr += dp[i - 1][j + 1];
                else
                    dr += billion;

                dp[i][j] = Math.min(up, Math.min(dl, dr));
            }
        }

        int mini = Integer.MAX_VALUE;
        for (int j = 0; j < m; j++) {
            mini = Math.min(mini, dp[n - 1][j]);
        }
        return mini;
    }

    public int minFallingPathSumSpaceOptimized(int[][] matrix) {
        int billion = 1_000_000_000;
        int n = matrix.length;
        int m = matrix[0].length;
        int[] prev = new int[m];
        for (int j = 0; j < m; j++) {
            prev[j] = matrix[0][j];
        }
        for (int i = 1; i < n; i++) {
            int[] curr = new int[m];
            for (int j = 0; j < m; j++) {
                int up = matrix[i][j] + prev[j];
                int dl = matrix[i][j], dr = matrix[i][j];
                if (j - 1 >= 0)
                    dl += prev[j - 1];
                else
                    dl += billion;

                if (j + 1 < m)
                    dr += prev[j + 1];
                else
                    dr += billion;

                curr[j] = Math.min(up, Math.min(dl, dr));
            }
            prev = curr;
        }

        int mini = Integer.MAX_VALUE;
        for (int j = 0; j < m; j++) {
            mini = Math.min(mini, prev[j]);
        }
        return mini;
    }

    public boolean isIsomorphic(String s, String t) {
        char[] sChaarcterMapping = new char[128];
        char[] tChaarcterMapping = new char[128];
        for (int i = 0; i < s.length(); i++) {
            char sChar = s.charAt(i);
            char tChar = t.charAt(i);
            if (sChaarcterMapping[(int) sChar] == '\u0000' && tChaarcterMapping[(int) tChar] == '\u0000') {
                sChaarcterMapping[(int) sChar] = tChar;
                tChaarcterMapping[(int) tChar] = sChar;
                continue;
            }
            if (sChaarcterMapping[(int) sChar] != tChar || tChaarcterMapping[(int) tChar] != sChar) {
                return false;
            }
        }
        return true;
    }

    public boolean canJump(int[] nums) {
        int currJump = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (currJump < 0) {
                return false;
            } else if (nums[i] > currJump) {
                currJump = nums[1];
            }
            currJump--;
        }
        return true;
    }

    public boolean dfsOnWordExist(char[][] board, String word, boolean[][] visited, int i, int j, int currentIdx) {
        if (currentIdx == word.length()) {
            return true;
        }

        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || visited[i][j]
                || board[i][j] != word.charAt(currentIdx)) {
            return false;
        }
        visited[i][j] = true;
        boolean childValue = dfsOnWordExist(board, word, visited, i + 1, j, currentIdx + 1) ||
                dfsOnWordExist(board, word, visited, i - 1, j, currentIdx + 1) ||
                dfsOnWordExist(board, word, visited, i, j + 1, currentIdx + 1)
                || dfsOnWordExist(board, word, visited, i, j - 1, currentIdx + 1);
        visited[i][j] = false;
        return childValue;
    }

    public boolean existWordInGrid(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == word.charAt(0)) {
                    boolean result = dfsOnWordExist(board, word, visited, i, j, 0);
                    if (result) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public static int maximumChocolatesMemo(int i, int j1, int j2, int n, int m, int[][] grid, int[][][] dp) {
        if (j1 < 0 || j2 < 0 || j1 >= m || j2 >= m) {
            return (int) (Math.pow(-10, 9));
        }
        if (i == n - 1) {
            if (j1 == j2) {
                return grid[i][j1];
            } else {
                return grid[i][j1] + grid[i][j2];
            }
        }
        if (dp[i][j1][j2] != -1) {
            return dp[i][j1][j2];
        }
        int maxi = Integer.MIN_VALUE;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int ans;
                if (j1 == j2) {
                    ans = grid[i][j1]
                            + maximumChocolatesMemo(i + 1, j1 + di, j2 + dj, n, m, grid, dp);
                } else {
                    ans = grid[i][j1] + grid[i][j2]
                            + maximumChocolatesMemo(i + 1, j1 + di, j2 + dj, n, m, grid, dp);
                }
                maxi = Math.max(maxi, ans);
            }
        }
        return dp[i][j1][j2] = maxi;
    }

    public static int maximumChocolates(int n, int m, int[][] grid) {
        int[][] front = new int[m][m];

        for (int j1 = 0; j1 < m; j1++) {
            for (int j2 = 0; j2 < m; j2++) {
                if (j1 == j2) {
                    front[j1][j2] = grid[n - 1][j1];
                } else {
                    front[j1][j2] = grid[n - 1][j1] + grid[n - 1][j2];
                }
            }
        }

        for (int i = n - 2; i >= 0; i--) {
            int[][] curr = new int[m][m];
            for (int j1 = m - 1; j1 >= 0; j1--) {
                for (int j2 = m - 1; j2 >= 0; j2--) {
                    int maxi = Integer.MIN_VALUE;
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ans;
                            if (j1 == j2) {
                                ans = grid[i][j1];
                            } else {
                                ans = grid[i][j1] + grid[i][j2];
                            }
                            if (j1 + di < 0 || j2 + dj < 0 || j1 + di >= m || j2 + dj >= m) {
                                ans += Math.pow(-10, 9);
                            } else {
                                ans += front[j1 + di][j2 + dj];
                            }
                            maxi = Math.max(maxi, ans);
                        }
                    }
                    curr[j1][j2] = maxi;
                }
            }
            front = curr;
        }
        return front[0][m - 1];
    }

    public static boolean subsetSumToKRecursion(int idx, int target, int[] arr, int[][] dp) {
        if (target == 0) {
            return true;
        }
        if (idx == 0) {
            return arr[0] == target;
        }
        if (dp[idx][target] != -1) {
            return dp[idx][target] == 0 ? false : true;
        }

        boolean notPick = subsetSumToKRecursion(idx - 1, target, arr, dp);
        boolean pick = false;
        if (arr[idx] <= target) {
            pick = subsetSumToKRecursion(idx - 1, target - arr[idx], arr, dp);
        }
        return (dp[idx][target] = pick || notPick == true ? 1 : 0) == 0 ? false : true;
    }

    public int countStudents(int[] students, int[] sandwiches) {
        int n = students.length;
        int zeroes = 0, ones = 0;
        for (int student : students) {
            if (student == 0) {
                zeroes++;
            } else {
                ones++;
            }
        }
        int i = 0;
        while (i < n) {
            if (sandwiches[i] == 0) {
                if (zeroes > 0) {
                    zeroes--;
                } else {
                    break;
                }
            } else {
                if (ones > 0) {
                    ones--;
                } else {
                    break;
                }
            }
            i++;
        }
        return n - i;
    }

    public int timeRequiredToBuy(int[] tickets, int k) {
        int n = tickets.length;
        if (tickets[k] == 1) {
            return k + 1;
        }
        int timeSpent = 0;
        for (int i = 0; i < n; i++) {
            if (i > k && tickets[i] > tickets[k]) {
                timeSpent += tickets[k] - 1;
            } else {
                timeSpent += Math.min(tickets[i], tickets[k]);
            }
        }
        return timeSpent;
    }

    public int[] deckRevealedIncreasing(int[] deck) {
        int n = deck.length;
        Arrays.sort(deck);
        Queue<Integer> position = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            position.offer(i);
        }
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[position.poll()] = deck[i];
            if (!position.isEmpty()) {
                position.offer(position.poll());
            }
        }
        return result;
    }

    public String removeKdigits(String num, int k) {
        if (k == num.length()) {
            return "0";
        }
        // monotonic stack with increasing order
        Deque<Character> monotonicDequeue = new ArrayDeque<>();
        for (char digit : num.toCharArray()) {
            // while k > 0 and there is element in stack and stack top is greater then the
            // current number
            while (k > 0 && !monotonicDequeue.isEmpty() && digit < monotonicDequeue.peekLast()) {
                monotonicDequeue.removeLast();
                k--;
            }
            if (monotonicDequeue.isEmpty() && digit == '0') {
                continue;
            }
            monotonicDequeue.addLast(digit);
        }
        while (k > 0 && !monotonicDequeue.isEmpty()) {
            monotonicDequeue.removeLast();
            k--;
        }
        StringBuilder result = new StringBuilder();
        while (!monotonicDequeue.isEmpty()) {
            result.append(monotonicDequeue.pollFirst());
        }
        if (result.length() == 0) {
            return "0";
        }
        return result.toString();
    }

    public int trap(int[] height) {
        int n = height.length;
        int left = 0, right = n - 1;
        int maxLeft = 0, maxRight = 0;
        int traps = 0;
        while (left <= right) {
            if (height[left] <= height[right]) {
                maxLeft = Math.max(height[left], maxLeft);
                traps += maxLeft - height[left];
                left++;
            } else {
                maxRight = Math.max(height[right], maxRight);
                traps += maxRight - height[right];
                right--;
            }
        }
        return traps;
    }

    public static boolean subsetSumToKTabulation(int n, int k, int arr[]) {
        // Write your code here.
        boolean[][] dp = new boolean[n][k + 1];
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        if (arr[0] <= k) {
            dp[0][arr[0]] = true;
        }

        for (int ind = 1; ind < n; ind++) {
            for (int target = 1; target < k + 1; target++) {
                boolean pick = false;
                if (arr[ind] <= target) {
                    pick = dp[ind - 1][target - arr[ind]];
                }
                boolean notPick = dp[ind - 1][target];
                dp[ind][target] = pick || notPick;
            }
        }
        return dp[n][k + 1];
    }

    public static boolean subsetSumToKSpaceOptimization(int n, int k, int arr[]) {
        // Write your code here.
        boolean[] prev = new boolean[k + 1];

        prev[0] = true;
        if (arr[0] <= k) {
            prev[arr[0]] = true;
        }

        for (int ind = 1; ind < n; ind++) {
            boolean[] curr = new boolean[k + 1];
            curr[0] = true;
            for (int target = 1; target < k + 1; target++) {
                boolean pick = false;
                if (arr[ind] <= target) {
                    pick = prev[target - arr[ind]];
                }
                boolean notPick = prev[target];
                curr[target] = pick || notPick;
            }
            prev = curr;
        }
        return prev[k];
    }

    public int maxDepth(String s) {
        int n = s.length();
        int maxi = 0;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                ans++;
            }
            if (s.charAt(i) == ')') {
                ans--;
            }
            maxi = Math.max(maxi, ans);
        }
        if (ans == 0) {
            return maxi;
        }
        return 0;
    }

    public String makeGood(String s) {
        Stack<Character> stack = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (!stack.isEmpty() && Math.abs(ch - stack.peek()) == 32) {
                stack.pop();
            } else {
                stack.add(ch);
            }
        }
        String results = "";
        // StringBuilder result = new StringBuilder();
        while (stack.isEmpty()) {
            // result.insert(0, stack.pop());
            results += stack.pop();
        }
        return results;
    }

    public String minRemoveToMakeValid(String s) {
        int open = 0, close = 0, flag = 0;
        int n = s.length();
        for (int i = 0; i < n; i++) {
            Character ch = s.charAt(i);
            if (ch == '(') {
                open++;
                flag++;
            } else if (ch == ')' && flag > 0) {
                close++;
                flag--;
            }
        }
        int validParanthesis = Math.min(open, close);
        open = validParanthesis;
        close = validParanthesis;
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < n; i++) {
            Character ch = s.charAt(i);
            if (ch == '(') {
                if (open > 0) {
                    result.append(ch);
                    open--;
                }
            } else if (ch == ')') {
                if (close > 0 && close > open) {
                    result.append(ch);
                    close--;
                }
            } else {
                result.append(ch);
            }
        }
        return result.toString();
    }

    public boolean checkValidString(String s) {
        int leftMax = 0, leftMin = 0;
        for (char ch : s.toCharArray()) {
            if (ch == '(') {
                leftMax++;
                leftMin++;
            } else if (ch == ')') {
                leftMax--;
                leftMin--;
            } else if (ch == '*') {
                leftMax++;
                leftMin--;
            }
            if (leftMax < 0) {
                return false;
            }
            leftMin = Math.max(leftMin, 0);
        }
        return leftMin == 0;
    }

    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        if (depth == 1) {
            TreeNode ptr = new TreeNode(val);
            ptr.left = root;
            return ptr;
        }
        Queue<TreeNode> queueBfs = new LinkedList<>();
        queueBfs.offer(root);
        int level = 1;
        while (!queueBfs.isEmpty() && level < depth) {
            int n = queueBfs.size();
            for (int i = 0; i < n; i++) {
                if (level == depth - 1) {
                    TreeNode temp = queueBfs.poll();
                    TreeNode next = temp.left;
                    temp.left = new TreeNode(val);
                    temp.left.left = next;

                    next = temp.right;
                    temp.right = new TreeNode(val);
                    temp.right.right = next;
                } else {
                    TreeNode curr = queueBfs.poll();
                    if (curr.left != null)
                        queueBfs.offer(curr.left);
                    if (curr.right != null)
                        queueBfs.offer(curr.right);
                }
            }
            level++;
        }
        return root;
    }

    public boolean subSetPartitionRecursion(int idx, int target, int[] nums, int[][] dp) {
        if (target == 0) {
            return true;
        }
        if (idx == 0) {
            return nums[0] == target;
        }
        if (dp[idx][target] != -1) {
            return dp[idx][target] == 0 ? false : true;
        }
        boolean notPick = subSetPartitionRecursion(idx - 1, target, nums, dp);
        boolean pick = false;
        if (nums[idx] <= target) {
            pick = subSetPartitionRecursion(idx - 1, target - nums[idx], nums, dp);
        }
        dp[idx][target] = pick || notPick == true ? 1 : 0;
        return pick || notPick;
    }

    public boolean canPartitionTabulation(int[] nums) {
        int totalSum = 0;
        int n = nums.length;
        for (int num : nums) {
            totalSum += num;
        }
        if (totalSum % 2 != 0) {
            return false;
        }
        totalSum = totalSum / 2;
        boolean[][] dp = new boolean[n][totalSum + 1];
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        if (nums[0] <= totalSum) {
            dp[0][nums[0]] = true;
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 0; target <= totalSum; target++) {
                boolean notPick = dp[idx - 1][target];
                boolean pick = false;
                if (nums[idx] <= target) {
                    pick = dp[idx - 1][target - nums[idx]];
                }
                dp[idx][target] = pick || notPick;
            }
        }

        return dp[n][totalSum];
    }

    public boolean canPartitionSpaceOptimization(int[] nums) {
        int totalSum = 0;
        int n = nums.length;
        for (int num : nums) {
            totalSum += num;
        }
        if (totalSum % 2 != 0) {
            return false;
        }
        totalSum = totalSum / 2;
        boolean[] prev = new boolean[totalSum + 1], curr = new boolean[totalSum + 1];
        prev[0] = true;
        curr[0] = true;
        if (nums[0] <= totalSum) {
            prev[nums[0]] = true;
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 1; target <= totalSum; target++) {
                boolean notPick = prev[target];
                boolean pick = false;
                if (nums[idx] <= target) {
                    pick = prev[target - nums[idx]];
                }
                curr[target] = pick || notPick;
            }
            prev = curr.clone();
        }

        return prev[totalSum];
    }

    public static int minSubsetSumDifference(int[] arr, int n) {
        // Write your code here.
        int totalSum = 0;
        for (int num : arr) {
            totalSum += num;
        }
        int arraySum = totalSum;
        totalSum = totalSum / 2;
        boolean[] prev = new boolean[totalSum + 1];
        boolean[] curr = new boolean[totalSum + 1];
        prev[0] = true;
        curr[0] = true;
        if (arr[0] <= totalSum) {
            prev[arr[0]] = true;
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 0; target < totalSum + 1; target++) {
                boolean notPick = prev[target];
                boolean pick = false;
                if (arr[idx] <= target) {
                    pick = prev[target - arr[idx]];
                }
                curr[target] = pick || notPick;
            }
            prev = curr.clone();
        }
        int minDifference = Integer.MAX_VALUE;
        for (int i = 0; i < totalSum + 1; i++) {
            if (prev[i]) {
                minDifference = Math.min(minDifference, Math.abs(arraySum - i - i));
            }
        }
        return minDifference;
    }

    public static int findWaysMemo(int idx, int target, int[] nums, int[][] dp) {
        if (idx == 0) {
            if (nums[0] == 0 && target == 0) {
                return 2;
            } else if (target == 0 || nums[0] == target) {
                return 1;
            } else {
                return 0;
            }
        }
        if (dp[idx][target] != -1) {
            return dp[idx][target];
        }
        int notPick = findWaysMemo(idx - 1, target, nums, dp);
        int pick = 0;
        if (nums[idx] <= target) {
            pick = findWaysMemo(idx - 1, target - nums[idx], nums, dp);
        }
        return dp[idx][target] = notPick + pick;
    }

    public static int findWaysTabulation(int num[], int tar) {
        int mod = 1_000_000_007;
        int n = num.length;
        int[] prev = new int[tar + 1];
        if (num[0] == 0) { // when target = 0 and num[0] == 0
            prev[0] = 2;
        } else {
            prev[0] = 1; // when target = 0 and nums[0] != 0
        }
        if (num[0] != 0 && num[0] <= tar) { // when nums[0] != 0 and target != 0 and nums[0] == target
            prev[num[0]] = 1;
        }
        for (int idx = 1; idx < n; idx++) {
            int[] curr = new int[tar + 1];
            for (int target = 0; target < tar + 1; target++) {
                int notPick = prev[target];
                int pick = 0;
                if (num[idx] <= target) {
                    pick = prev[target - num[idx]];
                }
                curr[target] = (pick + notPick) % mod;
            }
            prev = curr;
        }
        return prev[tar];
    }

    public static int countParticianMemo(int idx, int target, int[] arr, int[][] dp) {
        if (idx == 0) {
            if (arr[0] == 0 && target == 0) {
                return 2;
            } else if (arr[0] == target || target == 0) {
                return 1;
            } else {
                return 0;
            }
        }
        if (dp[idx][target] != -1) {
            return dp[idx][target];
        }
        int notPick = countParticianMemo(idx - 1, target, arr, dp);
        int pick = 0;
        if (arr[idx] <= target) {
            pick = countParticianMemo(idx - 1, target - arr[idx], arr, dp);
        }
        return dp[idx][target] = pick + notPick;
    }

    public static int countPartitionsRecursion(int n, int d, int[] arr) {
        int mod = 1_000_000_007;
        int totalSum = 0;
        for (int num : arr) {
            totalSum += num;
        }
        if (totalSum - d < 0) {
            return 0;
        }
        if ((totalSum - d) % 2 == 1) {
            return 0;
        }
        int s2 = (totalSum - d) / 2;
        int[] prev = new int[s2 + 1];
        if (arr[0] == 0) {
            prev[0] = 2;
        } else {
            prev[0] = 1;
        }
        if (arr[0] != 0 && arr[0] <= s2) {
            prev[arr[0]] = 1;
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = s2; target >= 0; target--) {
                int notPick = prev[target];
                int pick = 0;
                if (arr[idx] <= target) {
                    pick = prev[target - arr[idx]];
                }
                prev[target] = (pick + notPick) % mod;
            }
        }
        return prev[s2];
    }

    static int knapsackRecursionMemo(int idx, int remainWeight, int[] weight, int[] value, int[][] dp) {
        if (remainWeight == 0) {
            return 0;
        }
        if (idx == 0) {
            if (weight[idx] <= remainWeight) {
                return value[idx];
            } else {
                return 0;
            }
        }
        if (dp[idx][remainWeight] != -1) {
            return dp[idx][remainWeight];
        }
        int notPick = knapsackRecursionMemo(idx - 1, remainWeight, weight, value, dp);
        int pick = Integer.MIN_VALUE;
        if (weight[idx] <= remainWeight) {
            pick = value[idx] + knapsackRecursionMemo(idx - 1, remainWeight - weight[idx], weight, value, dp);
        }
        return dp[idx][remainWeight] = Math.max(pick, notPick);
    }

    static int knapsack(int[] weight, int[] value, int n, int maxWeight) {
        int[] prev = new int[maxWeight + 1];
        for (int i = weight[0]; i < maxWeight + 1; i++) {
            prev[i] = value[0];
        }
        for (int idx = 1; idx < n; idx++) {
            for (int currWeight = maxWeight; currWeight >= 0; currWeight--) {
                int notPick = prev[currWeight];
                int pick = Integer.MIN_VALUE;
                if (weight[idx] <= currWeight) {
                    pick = value[idx] + prev[currWeight - weight[idx]];
                }
                prev[currWeight] = Math.max(pick, notPick);
            }
        }
        return prev[maxWeight];
    }

    public static int minimumCoinsMemo(int idx, int remaningTarget, int[] nums, int[][] dp) {
        int maxValue = 1_000_000_000;
        if (remaningTarget == 0) {
            return 0;
        }
        if (idx == 0) {
            if (remaningTarget % nums[0] != 0) {
                return maxValue;
            } else {
                return remaningTarget / nums[0];
            }
        }
        if (dp[idx][remaningTarget] != -1) {
            return dp[idx][remaningTarget];
        }
        int notPick = minimumCoinsMemo(idx - 1, remaningTarget, nums, dp);
        int pick = maxValue;
        if (nums[idx] <= remaningTarget) {
            pick = 1 + minimumCoinsMemo(idx, remaningTarget - nums[idx], nums, dp);
        }
        return dp[idx][remaningTarget] = Math.min(notPick, pick);
    }

    public int coinChange(int[] coins, int amount) {
        int maxValue = 1_000_000_000;
        int n = coins.length;
        int[] prev = new int[amount + 1]; // bydefault target = 0 == 0
        for (int i = 0; i < amount + 1; i++) {
            if (i % coins[0] != 0) {
                prev[i] = maxValue;
            } else {
                prev[i] = i / coins[0];
            }
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 1; target < amount + 1; target++) { // as per recursion target == 0 is 0
                int notPick = prev[target];
                int pick = maxValue;
                if (coins[idx] <= target) {
                    pick = 1 + prev[target - coins[idx]];
                }
                prev[target] = Math.min(pick, notPick);
            }
        }
        int possible = prev[amount];
        if (possible >= maxValue) {
            return -1;
        }
        return possible;
    }

    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }
        if ((totalSum - target) % 2 != 0) {
            return 0;
        }
        int s2 = (totalSum - target) / 2;
        int[] prev = new int[s2 + 1];
        if (nums[0] == 0) {
            prev[0] = 2;
        } else {
            prev[0] = 1;
        }
        if (nums[0] != 0 && nums[0] <= s2) {
            prev[nums[0]] = 1;
        }
        for (int idx = 1; idx < n; idx++) {
            for (int tar = s2; tar >= 0; tar--) {
                int notPick = prev[tar];
                int pick = 0;
                if (nums[idx] <= tar) {
                    pick = prev[tar - nums[idx]];
                }
                prev[tar] = pick + notPick;
            }
        }
        return prev[s2];
    }

    public int coinChangePossibleRecursion(int idx, int remainAmount, int[] coins, int[][] dp) {
        if (remainAmount == 0) {
            return 1;
        }
        if (idx == 0) {
            if (remainAmount % coins[0] == 0) {
                return 1;
            } else {
                return 0;
            }
        }
        if (dp[idx][remainAmount] != -1) {
            return dp[idx][remainAmount];
        }
        int notPick = coinChangePossibleRecursion(idx - 1, remainAmount, coins, dp);
        int pick = 0;
        if (coins[idx] <= remainAmount) {
            pick = coinChangePossibleRecursion(idx, remainAmount - coins[idx], coins, dp);
        }
        return dp[idx][remainAmount] = pick + notPick;
    }

    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[] prev = new int[amount + 1];
        for (int i = 0; i < amount + 1; i++) {
            if (i % coins[0] == 0) {
                prev[i] = 1;
            }
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 0; target < amount + 1; target++) {
                int notPick = prev[target];
                int pick = 0;
                if (coins[idx] <= target) {
                    pick = prev[target - coins[idx]];
                }
                prev[target] = pick + notPick;
            }
        }
        return prev[amount];
    }

    public static int unboundedKnapsackSpaceOptimization(int n, int w, int[] profit, int[] weight) {
        int[] prev = new int[w + 1]; // value of w ==0 will be always 0
        for (int i = 0; i < w + 1; i++) {
            if (i % weight[0] == 0) {
                prev[i] = (i / weight[0]) * profit[0];
            }
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 0; target < w + 1; target++) {
                int notPick = prev[target];
                int pick = Integer.MIN_VALUE;
                if (weight[idx] <= target) {
                    pick = profit[idx] + prev[target - weight[idx]];
                }
                prev[target] = Math.max(pick, notPick);
            }
        }
        return prev[w];
    }

    public static int cutRodMemo(int idx, int remainingLength, int price[], int[][] dp) {
        if (remainingLength == 0) {
            return 0;
        }
        if (idx == 0) {
            return remainingLength * price[0];
        }
        if (dp[idx][remainingLength] != -1) {
            return dp[idx][remainingLength];
        }
        int notPick = cutRodMemo(idx - 1, remainingLength, price, dp);
        int pick = Integer.MIN_VALUE;
        if (idx + 1 <= remainingLength) {
            pick = price[idx] + cutRodMemo(idx, remainingLength - (idx + 1), price, dp);
        }
        return dp[idx][remainingLength] = Math.max(pick, notPick);
    }

    public static int cutRod(int price[], int n) {
        int[] prev = new int[n + 1];
        for (int i = 0; i < n + 1; i++) {
            prev[i] = i * price[0];
        }
        for (int idx = 1; idx < n; idx++) {
            for (int target = 0; target < n + 1; target++) {
                int notPick = prev[target];
                int pick = Integer.MIN_VALUE;
                if (idx + 1 <= target) {
                    pick = price[idx] + prev[target - (idx + 1)];
                }
                prev[target] = Math.max(pick, notPick);
            }
        }
        return prev[n];
    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        Stack<Integer> monotonicIncreasing = new Stack<>();
        int[] leftSmall = new int[n], rightSmall = new int[n];
        for (int i = 0; i < n; i++) {
            while (!monotonicIncreasing.isEmpty() && heights[monotonicIncreasing.peek()] >= heights[i]) {
                monotonicIncreasing.pop();
            }
            if (monotonicIncreasing.isEmpty()) {
                leftSmall[i] = 0;
            } else {
                leftSmall[i] = monotonicIncreasing.peek() + 1;
            }
            monotonicIncreasing.add(i);
        }

        // clearing the stack
        while (!monotonicIncreasing.isEmpty()) {
            monotonicIncreasing.pop();
        }
        for (int i = n - 1; i >= 0; i--) {
            while (!monotonicIncreasing.isEmpty() && heights[monotonicIncreasing.peek()] >= heights[i]) {
                monotonicIncreasing.pop();
            }
            if (monotonicIncreasing.isEmpty()) {
                rightSmall[i] = n - 1;
            } else {
                rightSmall[i] = monotonicIncreasing.peek() - 1;
            }
            monotonicIncreasing.add(i);
        }
        int maxi = 0;
        for (int i = 0; i < n; i++) {
            int currArea = (rightSmall[i] - leftSmall[i] + 1) * heights[i];
            maxi = Math.max(maxi, currArea);
        }
        return maxi;
    }

    public int largestRectangleAreaOptimized(int[] heights) {
        int n = heights.length;
        Stack<Integer> monotonicIncreasing = new Stack<>();
        int maxi = 0;
        for (int i = 0; i <= n; i++) {
            while (!monotonicIncreasing.isEmpty() && (i == n || heights[monotonicIncreasing.peek()] >= heights[i])) {
                int height = heights[monotonicIncreasing.pop()];
                int width;
                if (monotonicIncreasing.isEmpty()) {
                    width = i;
                } else {
                    width = i - monotonicIncreasing.peek() - 1;
                }
                maxi = Math.max(maxi, width * height);
            }
            monotonicIncreasing.add(i);
        }
        return maxi;
    }

    public int maximalRectangle(char[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        int[] height = new int[col];
        int maxi = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 1) {
                    height[j]++;
                } else {
                    height[j] = 0;
                }
            }
            int currArea = largestRectangleAreaOptimized(height);
            maxi = Math.max(maxi, currArea);
        }
        return maxi;
    }

    public int countSquares(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[] prev = new int[m];
        int count = 0;
        for (int j = 0; j < m; j++) {
            prev[j] = matrix[0][j];
            count += prev[j];
        }
        for (int i = 1; i < n; i++) {
            int[] curr = new int[m];
            curr[0] = matrix[i][0];
            count += curr[0];
            for (int j = 1; j < m; j++) {
                if (matrix[i][j] == 1) {
                    curr[j] = 1 + Math.min(prev[j - 1], Math.min(prev[j], curr[j - 1]));
                } else {
                    curr[j] = 0;
                }
                count += curr[j];
            }
            prev = curr;
        }
        // int count = 0;
        // for (int i = 0; i < n; i++) {
        // for (int j = 0; j < m; j++) {
        // count += dp[i][j];
        // }
        // }
        return count;
    }

    public int sumOfNumbersDfs(TreeNode root, int pathSum) {
        if (root == null) {
            return 0;
        }
        pathSum = pathSum * 10 + root.val;
        if (root.left == null && root.right == null) {
            return pathSum;
        }
        return sumOfNumbersDfs(root.left, pathSum) + sumOfNumbersDfs(root.right, pathSum);
    }

    public int sumNumbers(TreeNode root) {
        return sumOfNumbersDfs(root, 0);
    }

    public int longestCommonSubsequenceMemo(int i, int j, String s1, String s2, int[][] dp) {
        if (i == 0 || j == 0) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            return dp[i][j] = 1 + longestCommonSubsequenceMemo(i - 1, j - 1, s1, s2, dp);
        } else {
            return dp[i][j] = Math.max(longestCommonSubsequenceMemo(i - 1, j, s1, s2, dp),
                    longestCommonSubsequenceMemo(i - 1, j, s1, s2, dp));
        }
    }

    public int longestCommonSubsequence(String s1, String s2) {
        int n = s1.length();
        int m = s2.length();
        int[] prev = new int[m + 1];

        // byDefault will be zero so removing the loop for making it to zero
        // j == 0 || i == 0
        for (int i = 1; i < n + 1; i++) {
            int[] curr = new int[m + 1];
            for (int j = 1; j < m + 1; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    curr[j] = Math.max(prev[j], curr[j - 1]);
                }
            }
        }
        return prev[m];
    }

    public static String findLCS(int n, int m, String s1, String s2) {
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        int i = n, j = m;
        String result = "";
        while (i > 0 && j > 0) {
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                result = s1.charAt(i - 1) + result;
            } else {
                if (dp[i - 1][j] < dp[i][j - 1]) {
                    j--;
                } else {
                    i--;
                }
            }
        }
        return result;
    }

    public static int lcs(String s1, String s2) {
        int n = s1.length();
        int m = s2.length();
        int[] prev = new int[m + 1];
        int[] curr = new int[m + 1];
        int maxi = 0;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    curr[j] = 1 + prev[j - 1];
                    maxi = Math.max(maxi, curr[j]);
                } else {
                    curr[j] = 0;
                }
            }
            prev = curr.clone();
        }
        return maxi;
    }

    public int longestPalindromeSubseqMemo(int i, int j, String s1, String s2, int[][] dp) {
        if (i == 0 || j == 0) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            return dp[i][j] = 1 + longestPalindromeSubseqMemo(i - 1, j - 1, s1, s2, dp);
        } else {
            return dp[i][j] = Math.max(longestPalindromeSubseqMemo(i - 1, j, s1, s2, dp),
                    longestPalindromeSubseqMemo(i, j - 1, s1, s2, dp));
        }
    }

    public int longestPalindromeSubseq(String s) {
        String t = new StringBuilder(s).reverse().toString();
        int n = s.length();
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    curr[j] = Math.max(prev[j], curr[j - 1]);
                }
            }
            prev = curr.clone();
        }
        return prev[n];
    }

    public int minInsertionsRecursion(int i, int j, String s1, String s2, int[][] dp) {
        if (i == 0 || j == 0) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            return dp[i][j] = 1 + minInsertionsRecursion(i - 1, j - 1, s1, s2, dp);
        }
        return dp[i][j] = Math.max(minInsertionsRecursion(i - 1, j, s1, s2, dp),
                minInsertionsRecursion(i, j - 1, s1, s2, dp));
    }

    public int minInsertions(String s) {
        int n = s.length();
        String t = new StringBuilder(s).reverse().toString();
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    curr[j] = Math.max(prev[j], curr[j - 1]);
                }
            }
            prev = curr.clone();
        }
        return n - prev[n];
    }

    String smallestString = null;

    public String smallestFromLeaf(TreeNode root) {
        dfsSmallestString(root, new StringBuilder());
        return smallestString;
    }

    private void dfsSmallestString(TreeNode root, StringBuilder currString) {
        if (root == null) {
            return;
        }
        currString.insert(0, (char) ('a' + root.val));
        if (root.left == null && root.right == null) {
            updateSmallestString(currString.toString());
        } else {
            dfsSmallestString(root.left, currString);
            dfsSmallestString(root.right, currString);
        }
        currString.deleteCharAt(0);
    }

    private void updateSmallestString(String currString) {
        if (smallestString == null || currString.compareTo(smallestString) < 0) {
            smallestString = currString;
        }
    }

    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        int[] prev = new int[m + 1];
        int[] curr = new int[m + 1];
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    curr[j] = Math.max(prev[j], curr[j - 1]);
                }
            }
            prev = curr.clone();
        }
        return n + m - 2 * prev[m];
    }

    public int dfsIslandPerimeter(int currRow, int currCol, int[][] grid) {
        if (currRow < 0 || currRow >= grid.length || currCol < 0 || currCol >= grid[0].length
                || grid[currRow][currCol] == 0) {
            return 1;
        }
        if (grid[currRow][currCol] == -1) {
            return 0;
        }
        grid[currRow][currCol] = -1;
        return dfsIslandPerimeter(currRow - 1, currCol, grid) + dfsIslandPerimeter(currRow + 1, currCol, grid)
                + dfsIslandPerimeter(currRow, currCol - 1, grid) + dfsIslandPerimeter(currRow, currCol + 1, grid);
    }

    public int islandPerimeter(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    return dfsIslandPerimeter(row, col, grid);
                }
            }
        }
        return 0;
    }

    public int numIslands(char[][] grid) {
        int n = grid.length;
        int m = grid[0].length;

        int numOfIslands = 0;
        Queue<NumOfIslands> queueBfs = new LinkedList<>();
        int maxSize = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == '1') {
                    numOfIslands++;
                    queueBfs.offer(new NumOfIslands(i, j));
                    while (!queueBfs.isEmpty()) {
                        maxSize = Math.max(maxSize, queueBfs.size());
                        NumOfIslands curr = queueBfs.poll();
                        int[][] dxy = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
                        if (grid[curr.row][curr.col] != '1') {
                            continue;
                        }
                        grid[curr.row][curr.col] = '0';
                        for (int k = 0; k < 4; k++) {
                            int nextRow = curr.row + dxy[k][0];
                            int nextCol = curr.col + dxy[k][1];
                            if (nextRow >= 0 && nextRow < n && nextCol >= 0 && nextCol < m
                                    && grid[nextRow][nextCol] == '1') {
                                queueBfs.offer(new NumOfIslands(nextRow, nextCol));
                            }
                        }
                    }
                }
            }
        }
        return numOfIslands;
    }

    public int[] findFarmLandItetrative(int[][] land, int row, int col, int n, int m) {
        int[] coordinates = new int[4];
        coordinates[0] = row;
        coordinates[1] = col;
        int lastRow = row;
        int lastCol = col;
        while (lastRow < n && land[lastRow][col] == 1) {
            lastRow++;
        }
        while (lastCol < m && land[row][lastCol] == 1) {
            lastCol++;
        }
        coordinates[2] = lastRow;
        coordinates[3] = lastCol;

        for (int i = row; i <= lastRow; i++) {
            for (int j = col; j <= lastCol; j++) {
                land[i][j] = 0;
            }
        }
        return coordinates;
    }

    public int[][] findFarmland(int[][] land) {
        int n = land.length;
        int m = land[0].length;
        LinkedList<int[]> result = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (land[i][j] == 1) {
                    result.add(findFarmLandItetrative(land, i, j, n, m));
                }
            }
        }
        return result.toArray(new int[result.size()][]);
    }

    public int shortestCommonSupersequenceMemo(int i, int j, String s1, String s2, int[][] dp) {
        if (i == 0 || j == 0) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            return dp[i][j] = 1 + shortestCommonSupersequenceMemo(i - 1, j - 1, s1, s2, dp);
        }
        return dp[i][j] = Math.max(shortestCommonSupersequenceMemo(i - 1, j, s1, s2, dp),
                shortestCommonSupersequenceMemo(i, j - 1, s1, s2, dp));
    }

    public String shortestCommonSupersequence(String str1, String str2) {
        int n = str1.length();
        int m = str2.length();
        int[][] dp = new int[n + 1][m + 1];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        int i = n;
        int j = m;
        String ans = "";
        while (i > 0 && j > 0) {
            if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                ans += str1.charAt(i - 1);
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                ans += str1.charAt(i - 1);
                i--;
            } else {
                ans += str2.charAt(j - 1);
                j--;
            }
        }
        while (i > 0) {
            ans += str1.charAt(i - 1);
            i--;
        }
        while (j > 0) {
            ans += str2.charAt(j - 1);
            j--;
        }
        String finalAns = new StringBuilder(ans).reverse().toString();
        return finalAns;
    }

    public boolean dfsValidPath(int currNode, int destination, Map<Integer, List<Integer>> graph, boolean[] visited) {
        if (currNode == destination) {
            return true;
        }
        if (visited[currNode] == true) {
            return false;
        }
        visited[currNode] = true;
        for (int neighbour : graph.getOrDefault(currNode, new ArrayList<>())) {
            if (!visited[neighbour]) {
                if (dfsValidPath(neighbour, destination, graph, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean validPath(int n, int[][] edges, int source, int destination) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.computeIfAbsent(u, keyName -> new ArrayList<>()).add(v);
            graph.computeIfAbsent(v, keyName -> new ArrayList<>()).add(u);
        }
        boolean[] visited = new boolean[n];
        return dfsValidPath(source, destination, graph, visited);
    }

    public int numDistinctMemo(int i, int j, String s1, String s2, int[][] dp) {
        if (j == 0) {
            return 1;
        }
        if (i == 0) {
            return 0;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }

        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            return dp[i][j] = numDistinctMemo(i - 1, j - 1, s1, s2, dp) + numDistinctMemo(i - 1, j, s1, s2, dp);
        }
        return dp[i][j] = numDistinctMemo(i - 1, j, s1, s2, dp);
    }

    public int numDistinct(String s, String t) {
        int n = s.length();
        int m = t.length();
        int[] prev = new int[m + 1];
        prev[0] = 1;
        // below code is not required as its byDefault 0
        // for (int j = 1; j < m + 1; j++) {
        // dp[0][j] = 0;
        // }
        for (int i = 1; i < n + 1; i++) {
            for (int j = m; j > 0; j--) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    prev[j] = prev[j - 1] + prev[j];
                } else {
                    prev[j] = prev[j];
                }
            }
        }
        return prev[m];
    }

    public int minDistanceRecursion(int i, int j, String s1, String s2, int[][] dp) {
        if (j == 0) {
            return i;
        }
        if (i == 0) {
            return j;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
            return dp[i][j] = minDistanceRecursion(i - 1, j - 1, s1, s2, dp);
        }
        return dp[i][j] = 1 + Math.min(minDistanceRecursion(i - 1, j, s1, s2, dp), // delete
                Math.min(minDistanceRecursion(i, j - 1, s1, s2, dp), // insert
                        minDistanceRecursion(i - 1, j - 1, s1, s2, dp))); // replace
    }

    public int editMinDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        int[] prev = new int[m + 1];
        int[] curr = new int[m + 1];
        prev[0] = 0;
        for (int j = 0; j < m + 1; j++) {
            prev[j] = j;
        }
        for (int i = 1; i < n + 1; i++) {
            curr[0] = i;
            for (int j = 1; j < m + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    curr[j] = prev[j - 1];
                } else {
                    curr[j] = 1 + Math.min(prev[j], // delete
                            Math.min(curr[j - 1], // insert
                                    prev[j - 1])); // replace
                }
            }
        }
        return prev[m];
    }

    public int openLock(String[] deadends, String target) {
        Set<String> deadendSet = new HashSet<>(Arrays.asList(deadends));
        if (deadendSet.contains("0000") || deadendSet.contains(target)) {
            return -1;
        }
        if (target.equals("0000")) {
            return 0;
        }
        Queue<OpenLockPair> queueBfs = new LinkedList<>();
        queueBfs.offer(new OpenLockPair("0000", 0));
        Set<String> visited = new HashSet<>();
        visited.add("0000");
        while (!queueBfs.isEmpty()) {
            OpenLockPair currSequence = queueBfs.poll();
            for (int i = 0; i < 4; i++) {
                for (int delta : new int[] { -1, 1 }) {

                    int newChar = ((currSequence.value.charAt(i) - '0' + delta + 10) % 10);
                    String newSequence = currSequence.value.substring(0, i) + newChar
                            + currSequence.value.substring(i + 1);
                    if (!visited.contains(newSequence) && !deadendSet.contains(newSequence)) {
                        if (newSequence.equals(target)) {
                            return currSequence.moves + 1;
                        }
                        visited.add(newSequence);
                        queueBfs.offer(new OpenLockPair(newSequence, currSequence.moves + 1));
                    }
                }
            }
        }
        return -1;
    }

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1)
            return Collections.singletonList(0);
        int[] degree = new int[n];
        Map<Integer, List<Integer>> adjanceListMap = new HashMap<>();
        for (int[] edge : edges) {
            degree[edge[0]]++;
            degree[edge[1]]++;
            adjanceListMap.computeIfAbsent(edge[0], x -> new ArrayList<>()).add(edge[1]);
            adjanceListMap.computeIfAbsent(edge[1], x -> new ArrayList<>()).add(edge[0]);
        }
        Queue<Integer> traversalBfs = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (degree[i] == 1) {
                traversalBfs.offer(i);
            }
        }
        List<Integer> result = new ArrayList<>();
        int processed = 0;
        while (!traversalBfs.isEmpty()) {
            int size = traversalBfs.size();
            processed += size;
            while (size > 0) {
                size--;
                int currNode = traversalBfs.poll();
                if (processed == n) {
                    result.add(currNode);
                }
                for (int adj : adjanceListMap.getOrDefault(currNode, new ArrayList<>())) {
                    if (--degree[adj] == 1) {
                        traversalBfs.offer(adj);
                    }
                }
            }
        }
        return result;
    }

    public class OpenLockPair {
        String value;
        int moves;

        public OpenLockPair(String value, int moves) {
            this.value = value;
            this.moves = moves;
        }

    }

    public int tribonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        int n1 = 0;
        int n2 = 1;
        int n3 = 1;
        for (int i = 3; i <= n; i++) {
            int ans = n1 + n2 + n3;
            n1 = n2;
            n2 = n3;
            n3 = ans;
        }
        return n3;
    }

    public int isAllStar(String s, int idx) {
        for (int i = 1; i < idx; i++) {
            if (s.charAt(i - 1) != '*') {
                return 0;
            }
        }
        return 1;
    }

    public int isMathcRecursion(int i, int j, String s1, String s2, int[][] dp) {
        if (i == 0 && j == 0) {
            return 1;
        }
        if (i == 0) {
            return 0;
        }
        if (j == 0) {
            return isAllStar(s1, i);
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        if (s1.charAt(i - 1) == s2.charAt(j - 1) || s1.charAt(i - 1) == '?') {
            return dp[i][j] = isMathcRecursion(i - 1, j - 1, s1, s2, dp);
        } else if (s1.charAt(i - 1) == '*') {
            return dp[i][j] = isMathcRecursion(i - 1, j, s1, s2, dp) == 1 || isMathcRecursion(i, j - 1, s1, s2, dp) == 1
                    ? 1
                    : 0;
        } else {
            return dp[i][j] = 0;
        }
    }

    public boolean isMatch(String s, String p) {
        int n = p.length();
        int m = s.length();
        boolean[] prev = new boolean[m + 1];
        boolean[] curr = new boolean[m + 1];
        prev[0] = true;
        // below for loop not required as it is false only bydefault
        // for (int j = 1; j < m + 1; j++) {
        // prev[j] = false;
        // }
        for (int i = 1; i < n + 1; i++) {
            if (p.charAt(i - 1) == '*' && prev[0] == true) {
                curr[0] = true;
            } else {
                curr[0] = false;
            }
            for (int j = 1; j < m + 1; j++) {
                if (p.charAt(i - 1) == s.charAt(j - 1) || p.charAt(i - 1) == '?') {
                    curr[j] = prev[j - 1];
                } else if (p.charAt(i - 1) == '*') {
                    curr[j] = prev[j] || curr[j - 1];
                } else {
                    curr[j] = false;
                }
            }
            prev = curr.clone();
        }
        return prev[m];
    }

    public int longestIdealStringRecursion(int idx, int prevOrder, String s, int k) {
        if (idx == 0) {
            return 0;
        }
        int notPick = longestIdealStringRecursion(idx - 1, prevOrder, s, k);
        int pick = 0;
        if (prevOrder == 0 || (s.charAt(idx - 1) - 'a') + 1 - prevOrder <= k) {
            pick = 1 + longestIdealStringRecursion(idx - 1, s.charAt(idx - 1) - 'a' + 1, s, k);
        }
        return Math.max(notPick, pick);
    }

    public int longestIdealString(String s, int k) {
        int n = s.length();
        int[] prev = new int[27];
        int[] curr = new int[27];
        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j <= 26; j++) {
                int notPick = prev[j];
                int pick = 0;
                if (j == 0 || Math.abs(s.charAt(i - 1) - 'a' + 1 - j) <= k) {
                    pick = 1 + prev[s.charAt(i - 1) - 'a' + 1];
                }
                curr[j] = Math.max(notPick, pick);
            }
            prev = curr.clone();
        }
        return prev[0];
    }

    public int minFallingPathSumRecursion(int i, int jPrevSelected, int m, int[][] grid, int[][] dp) {
        int mod = 1_000_000_000;
        if (i == 0) {
            return 0;
        }
        if (dp[i][jPrevSelected] != -1) {
            return dp[i][jPrevSelected];
        }
        int minOfAllNode = mod;
        for (int j = 0; j < m; j++) {
            if (j != jPrevSelected) {
                int currValue = grid[i - 1][j] + minFallingPathSumRecursion(i - 1, j, m, grid, dp);
                minOfAllNode = Math.min(minOfAllNode, currValue);
            }
        }
        return dp[i][jPrevSelected] = minOfAllNode;
    }

    public int minFallingPathSum(int[][] grid) {
        int mod = 1_000_000_000;
        int n = grid.length;
        int m = grid[0].length;
        int[] prev = new int[m + 1];
        int[] curr = new int[m + 1];
        for (int i = 1; i < n + 1; i++) {
            for (int jPrevSelected = 0; jPrevSelected <= m; jPrevSelected++) {
                int minOfAll = mod;
                for (int j = 0; j < m; j++) {
                    if (j != jPrevSelected) {
                        int currValue = grid[i - 1][j] + prev[j];
                        minOfAll = Math.min(minOfAll, currValue);
                    }
                }
                curr[jPrevSelected] = minOfAll;
            }
            prev = curr.clone();
        }
        return prev[m];
    }

    public int findRotateStepsRecursin(int i, int j, String ring, String key) {
        int mod = 1_000_000_000;
        if (j == key.length()) {
            return 0;
        }

        int minSteps = mod;
        for (int k = 0; k < ring.length(); k++) {
            if (ring.charAt(k) == key.charAt(j)) {
                int currSteps = Math.abs(i - k);
                minSteps = Math.min(minSteps, currSteps + 1 + findRotateStepsRecursin(k, j + 1, ring, key));
            }
        }
        return minSteps;
    }

    public int findRotateSteps(String ring, String key) {
        int mod = 1_000_000_000;
        int n = ring.length();
        int m = key.length();
        int[] ahead = new int[n];
        int[] curr = new int[n];
        for (int j = m - 1; j >= 0; j--) {
            for (int i = n - 1; i >= 0; i--) {
                int minSteps = mod;
                for (int k = 0; k < n; k++) {
                    if (ring.charAt(k) == key.charAt(j)) {
                        int currSteps = Math.abs(i - k);
                        minSteps = Math.min(minSteps, currSteps + 1 + ahead[k]);
                    }
                }
                curr[i] = minSteps;
            }
            ahead = curr.clone();
        }
        return ahead[0];
    }

    public void dfsSumOfDistancesInTree(Map<Integer, List<Integer>> graph, int u, int parent, int[] distFromRoot,
            int[] sizeOfCurrNode) {
        sizeOfCurrNode[u] = 1;
        for (int adjNode : graph.getOrDefault(u, new ArrayList<>())) {
            if (adjNode != parent) {
                distFromRoot[adjNode] = distFromRoot[u] + 1;
                dfsSumOfDistancesInTree(graph, adjNode, u, distFromRoot, sizeOfCurrNode);
                sizeOfCurrNode[u] += sizeOfCurrNode[adjNode];
            }
        }
        return;
    }

    public void dfsSumOfDistancesInTreeCurrNode(Map<Integer, List<Integer>> graph, int sdCurrNode, int u, int parent,
            int[] sizeOfCurrNode,
            int[] answer, int n) {
        answer[u] = sdCurrNode;
        for (int adjNode : graph.getOrDefault(u, new ArrayList<>())) {
            if (adjNode != parent) {
                int adjSd = sdCurrNode - sizeOfCurrNode[adjNode] + (n - sizeOfCurrNode[adjNode]);
                dfsSumOfDistancesInTreeCurrNode(graph, adjSd, adjNode, u, sizeOfCurrNode, answer, n);
            }
        }
        return;
    }

    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.computeIfAbsent(u, x -> new ArrayList<>()).add(v);
            graph.computeIfAbsent(v, x -> new ArrayList<>()).add(u);
        }
        int[] distFromRoot = new int[n];
        int[] sizeOfCurrNode = new int[n];
        dfsSumOfDistancesInTree(graph, 0, -1, distFromRoot, sizeOfCurrNode);
        int sumFromRoot = 0;
        for (int d : distFromRoot) {
            sumFromRoot += d;
        }
        int[] answer = new int[n];
        dfsSumOfDistancesInTreeCurrNode(graph, sumFromRoot, 0, -1, sizeOfCurrNode, answer, n);
        return answer;
    }

    public int minOperations(int[] nums, int k) {
        int totalXor = 0;
        for (int num : nums) {
            totalXor ^= num;
        }

        totalXor ^= k;
        int result = 0;
        while (totalXor > 0) {
            if ((totalXor & 1) == 1) {
                result++;
            }
            totalXor >>= 1;

        }
        return result;
    }

    public static int maximumProfit(ArrayList<Integer> prices) {
        // Write your code here.
        int profit = 0;
        int minValue = prices.get(0);
        for (int i = 1; i < prices.size(); i++) {
            minValue = Math.min(minValue, prices.get(i));
            profit = Math.max(profit, prices.get(i) - minValue);
        }
        return profit;
    }

    public int maxProfitMemo(int idx, int buy, int[] prices, int n, int[][] dp) {
        if (idx == n) {
            return 0;
        }
        if (dp[idx][buy] != -1) {
            return dp[idx][buy];
        }
        if (buy == 1) {
            return dp[idx][buy] = Math.max(-prices[idx] + maxProfitMemo(idx + 1, 0, prices, n, dp),
                    0 + maxProfitMemo(idx + 1, 1, prices, n, dp));
        } else {
            return dp[idx][buy] = Math.max(prices[idx] + maxProfitMemo(idx + 1, 1, prices, n, dp),
                    0 + maxProfitMemo(idx + 1, 0, prices, n, dp));
        }
    }

    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] ahead = new int[2];
        int[] curr = new int[2];
        ahead[0] = ahead[1] = 0;
        for (int i = n - 1; i >= 0; i--) {
            curr[1] = Math.max(-prices[i] + ahead[0],
                    0 + ahead[1]);

            curr[0] = Math.max(prices[i] + ahead[1],
                    0 + ahead[0]);
            ahead = curr.clone();
        }
        return ahead[1];
    }

    public int maxProfitAtMostK(int idx, int buy, int cap, int[] prices, int n) {
        if (cap == 0) {
            return 0;
        }
        if (idx == n) {
            return 0;
        }
        if (buy == 1) {
            return Math.max(-prices[idx] + maxProfitAtMostK(idx + 1, 0, cap, prices, n),
                    0 + maxProfitAtMostK(idx + 1, 1, cap, prices, n));
        } else {
            return Math.max(prices[idx] + maxProfitAtMostK(idx + 1, 1, cap - 1, prices, n),
                    0 + maxProfitAtMostK(idx + 1, 0, cap, prices, n));
        }
    }

    // public int maxProfitAtMost(int[] prices) {
    // int n = prices.length;
    // int[][][]
    // return maxProfitAtMostK(0, 1, 2, prices, n);
    // }

    public static void main(String[] args) {
        Solution solution = new Solution();
        // int[][] arr = { { -2147483646, -2147483645 }, { 2147483646, 2147483647 } };
        // int[] arr = { 1, 6, 11, 5 };
        // int output = solution.minDifference(arr, 4);
        char[][] array = {
                { '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '0', '1', '1' },
                { '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '0' },
                { '1', '0', '1', '1', '1', '0', '0', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '0', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1' },
                { '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '0', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '0', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1' },
                { '1', '0', '1', '1', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1', '1', '0', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '0' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '0', '0' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' }
        };

        System.out.println(solution.numIslands(array));
    }
}

class NumOfIslands {
    int row;
    int col;

    NumOfIslands(int row, int col) {
        this.row = row;
        this.col = col;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val) {
        this.val = val;
    }
}

class itemComparator implements Comparator<Item> {

    public int compare(Item a, Item b) {
        double r1 = (double) a.value / (double) a.weight;
        double r2 = (double) b.value / (double) b.weight;
        if (r1 < r2) {
            return 1;
        } else if (r1 > r2) {
            return -1;
        }
        return 0;
    }
}

class Job {
    int id, profit, deadline;

    Job(int x, int y, int z) {
        this.id = x;
        this.deadline = y;
        this.profit = z;
    }
}

class Item {
    int value, weight;

    Item(int x, int y) {
        this.value = x;
        this.weight = y;
    }
}