    void gauss_gpu_p(ele_t mat_p[N][N], int n) {
    LARGE_INTEGER nFreq;
    LARGE_INTEGER nBeginTime;
    LARGE_INTEGER nEndTime;
    double time;
    QueryPerformanceFrequency(&nFreq);
    queue q{ cpu_selector{} };
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
    // 分配共享内存
    ele_t(*new_mat)[N] = (ele_t(*)[N])malloc_shared<ele_t>(N * N, q);
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);
    QueryPerformanceCounter(&nBeginTime);
    for (int i = 0; i < n; i++) {
        // 列主元
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(new_mat[k][i]) > abs(new_mat[maxRow][i])) {
                maxRow = k;}
        }
        // 交换行
        if (maxRow != i) {
            for (int k = 0; k < n; k++) {
                std::swap(new_mat[i][k], new_mat[maxRow][k]);}
        }
        q.parallel_for(range{ n - (i + 1) }, [=](id<1> idx) {
            int j = idx[0] + i + 1;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            for (int k = i; k < n; k++)
                new_mat[j][k] -= new_mat[i][k] * div;
        }).wait();}
    QueryPerformanceCounter(&nEndTime);
    time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
    std::cout << time << ' ' << std::endl;
    if (n > 16) return;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << new_mat[i][j] << ' ';
        std::cout << std::endl;}
    std::cout << std::endl;}