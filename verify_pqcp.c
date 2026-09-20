#define _POSIX_C_SOURCE 200809L
/* Independent, read-only Project 2 verifier. Click Run on verify_pqcp.py,
 * or build directly as described in docs/c_verifier.md.
 * Bits 0/1 represent +1/-1. Count both u and L-u among nonzero shifts.
 * No search, pruning, approximate arithmetic, or external dependencies.
 */
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

/* Resource guard only, NOT a mathematical restriction on PQCP lengths. */
#define MAX_LENGTH 4096

typedef struct {
    unsigned fields;
    int bad;
    size_t line;
    long length, shifts[2], pacs;
    char *a, *b;
} Record;

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) ++s;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    return s;
}

static int integer(const char *s, long *value) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (s == end || errno == ERANGE || *end) return 0;
    *value = n;
    return 1;
}

static void error(Record *r, const char *message) {
    printf("  第 %zu 行起的紀錄：%s\n", r->line, message);
    r->bad = 1;
}

static int binary(const char *s, int length) {
    if (!s || strlen(s) != (size_t)length) return 0;
    return strspn(s, "01") == (size_t)length;
}

static int verify(Record *r, int length, size_t index, int *profile) {
    if (r->fields != 31) error(r, "缺少必要欄位（L、nonzero shifts、nonzero PACS、a、b）");
    if ((r->fields & 1) && r->length != length) error(r, "紀錄的 L 與輸入不符");
    int bits_ok = binary(r->a, length) && binary(r->b, length);
    if (!bits_ok) error(r, "a/b 必須各為 L 個 0/1，且長度相同");
    if (bits_ok) {
        int nonzero = 0, magnitude_ok = 1, positions[2] = {-1, -1};
        for (int u = 0; u < length; ++u) {
            int sum = 0;
            for (int i = 0; i < length; ++i) {
                int j = (i + u) % length;
                sum += (r->a[i] == r->a[j] ? 1 : -1);
                sum += (r->b[i] == r->b[j] ? 1 : -1);
            }
            profile[u] = sum;
            if (u && sum) {
                if (nonzero < 2) positions[nonzero] = u;
                ++nonzero;
                if (sum != 4 && sum != -4) magnitude_ok = 0;
            }
        }
        if (profile[0] != 2 * length) error(r, "S(0) 不等於 2L");
        if (nonzero != 2) error(r, "非零 shift 中不恰好有兩個非零 S(u)");
        if (!magnitude_ok) error(r, "非零 sidelobe 的絕對值不全是 4");
        for (int u = 1; u < length; ++u) {
            if (profile[u] != profile[length - u]) {
                error(r, "週期對稱性不符");
                break;
            }
        }
        if ((r->fields & 2) && (nonzero != 2 ||
            !((r->shifts[0] == positions[0] && r->shifts[1] == positions[1]) ||
              (r->shifts[0] == positions[1] && r->shifts[1] == positions[0]))))
            error(r, "記載的 nonzero shifts 與重新計算不符");
        if (r->fields & 4) {
            for (int u = 1; u < length; ++u) {
                if (profile[u] && profile[u] != r->pacs) {
                    error(r, "記載的 nonzero PACS 與重新計算不符");
                    break;
                }
            }
        }
        if (r->bad) {
            printf("  實際非零 sidelobes:");
            for (int u = 1; u < length; ++u)
                if (profile[u]) printf(" %d:%d", u, profile[u]);
            printf("\n");
        }
    }
    if (r->bad) printf("紀錄 #%zu：FAIL\n", index);
    return !r->bad;
}

static void clear_record(Record *r) {
    free(r->a);
    free(r->b);
    memset(r, 0, sizeof(*r));
}

int main(int argc, char **argv) {
    char input[128], filename[64];
    long parsed;
    if (argc > 3) {
        fprintf(stderr, "用法：%s [L [檔案路徑]]\n", argv[0]);
        return 2;
    }
    if (argc == 1) {
        printf("請輸入要驗證的長度 L：");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) return 2;
    }
    if (!integer(argc == 1 ? trim(input) : argv[1], &parsed) ||
        parsed < 1 || parsed > MAX_LENGTH) {
        fprintf(stderr, "L 必須是 1..%d 的整數（程式資源上限，非數學限制）。\n", MAX_LENGTH);
        return 2;
    }
    int length = (int)parsed;
    snprintf(filename, sizeof(filename), "%d.txt", length);
    const char *path = argc == 3 ? argv[2] : filename;
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        fprintf(stderr, "請在 PQCP2 專案根目錄執行，或指定檔案路徑。\n");
        return 2;
    }
    int *profile = calloc((size_t)length, sizeof(*profile));
    if (!profile) { fclose(file); return 2; }
    printf("驗證 L=%d，檔案：%s（唯讀）\n", length, path);
    Record record = {0};
    char *line = NULL;
    size_t capacity = 0, line_number = 0, total = 0, passed = 0;
    ssize_t size;
    int io_error = 0;
    while ((size = getline(&line, &capacity, file)) != -1) {
        ++line_number;
        if (memchr(line, '\0', (size_t)size)) {
            if (!record.line) record.line = line_number;
            error(&record, "檔案含 NUL 字元");
            continue;
        }
        char *s = trim(line);
        if (!*s) continue;
        if (!strncmp(s, "L=", 2) && record.line) {
            passed += (size_t)verify(&record, length, ++total, profile);
            clear_record(&record);
        }
        if (!record.line) record.line = line_number;
        unsigned field = 0;
        char *value = NULL;
        if (!strncmp(s, "L=", 2)) { field = 1; value = trim(s + 2); }
        else if (!strncmp(s, "nonzero shifts=", 15)) { field = 2; value = trim(s + 15); }
        else if (!strncmp(s, "nonzero PACS=", 13)) { field = 4; value = trim(s + 13); }
        else if (!strncmp(s, "a=", 2)) { field = 8; value = trim(s + 2); }
        else if (!strncmp(s, "b=", 2)) { field = 16; value = trim(s + 2); }
        else { error(&record, "未知欄位或格式錯誤"); continue; }
        if (record.fields & field) { error(&record, "欄位重複"); continue; }
        record.fields |= field;
        if (field == 1 || field == 4) {
            if (!integer(value, field == 1 ? &record.length : &record.pacs))
                error(&record, "整數欄位格式錯誤");
        } else if (field == 2) {
            char *comma = strchr(value, ',');
            if (!comma) error(&record, "nonzero shifts 必須列出兩個逗號分隔的整數");
            else {
                *comma = '\0';
                if (!integer(trim(value), &record.shifts[0]) ||
                    !integer(trim(comma + 1), &record.shifts[1]))
                    error(&record, "nonzero shifts 格式錯誤");
            }
        } else {
            char **destination = field == 8 ? &record.a : &record.b;
            *destination = strdup(value);
            if (!*destination) { io_error = 1; break; }
        }
    }
    if (!feof(file) || ferror(file)) io_error = 1;
    if (record.line) passed += (size_t)verify(&record, length, ++total, profile);
    clear_record(&record);
    free(line);
    free(profile);
    if (fclose(file) != 0) io_error = 1;
    printf("總紀錄：%zu\n通過：%zu\n不通過（含格式錯誤）：%zu\n", total, passed, total - passed);
    if (io_error) { fprintf(stderr, "讀取失敗或記憶體不足；未完成全部驗證。\n"); return 2; }
    if (!total) { fprintf(stderr, "檔案沒有任何紀錄，不能視為通過。\n"); return 1; }
    return passed == total ? 0 : 1;
}
