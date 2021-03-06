/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef LNORM_HPP
#define LNORM_HPP

#include <assert.h>
#include <limits.h>
#include <numeric>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

#include "bnorm/bnorm.hpp"

namespace lnorm {

using check_alg_t = bnorm::check_alg_t;
using flags_t = bnorm::flags_t;
const flags_t GLOB_STATS = bnorm::GLOB_STATS;
const flags_t USE_SCALESHIFT = bnorm::USE_SCALESHIFT;
const auto flags2str = bnorm::flags2str;
flags_t str2flags(const char *str);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    dims_t dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx}, stat_tag {tag::any};
    std::vector<flags_t> flags {0};
    std::vector<bool> inplace {true};
    check_alg_t check_alg = check_alg_t::ALG_AUTO;
    attr_t attr = {};
    bool allow_unimpl = false;
    const char *pattern = NULL;

    const char *perf_template_csv
            = "perf,%engine%,%dir%,%dt%,%tag%,%stat_tag%,%flags%,%DESC%,"
              "%Gops%,%-time%,%-Gbw%,%0time%,%0Gbw%";
    const char *perf_template_def
            = "perf,%engine%,%prb%,%Gops%,%-time%,%-Gbw%,%0time%,%0Gbw%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const dims_t &dims, const std::string &tag,
            const std::string &stat_tag, dir_t dir, dnnl_data_type_t dt,
            flags_t flags, bool inplace, const attr_t &attr,
            check_alg_t check_alg)
        : check_alg(check_alg)
        , dims(dims)
        , tag(tag)
        , stat_tag(stat_tag)
        , dir(dir)
        , dt(dt)
        , flags(flags)
        , inplace(inplace)
        , attr(attr)
        , ops(0)
        , ndims((int)dims.size()) {
        n = std::accumulate(
                dims.begin(), dims.end() - 1, 1, std::multiplies<int64_t>());
        c = dims[ndims - 1];
        eps = 1.f / 16;
        count_ops();
    }
    ~prb_t() {}

    check_alg_t check_alg;
    int64_t n, c;
    dims_t dims;
    std::string tag, stat_tag;
    dir_t dir;
    dnnl_data_type_t dt;
    flags_t flags;
    bool inplace;
    attr_t attr;
    float eps;
    double ops;
    int ndims;

    void count_ops() {
        if (ops > 0) return;
        bool use_scaleshift = flags & USE_SCALESHIFT;
        if (dir & FLAG_FWD) {
            ops = sizeof_dt(dt)
                    * ((2 - inplace) * n * c + 2 * n + use_scaleshift * 2 * c);
        } else {
            ops = sizeof_dt(dt)
                    * ((3 - inplace) * n * c + 2 * n + use_scaleshift * 2 * c
                            + 2 * c);
        }
    };
};

std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    virtual void dump_desc(std::ostream &s) const override { s << p_->dims; }

    virtual void dump_desc_csv(std::ostream &s) const override {
        s << p_->dims;
    }

    virtual void dump_flags(std::ostream &s) const override {
        s << flags2str(p_->flags);
    }

    virtual double ops() const override { return p_->ops; }
    virtual const attr_t *attr() const override { return &p_->attr; }
    virtual const dir_t *dir() const override { return &p_->dir; }
    virtual const dnnl_data_type_t *dt() const override { return &p_->dt; }
    virtual const std::string *tag() const override { return &p_->tag; }
    virtual const std::string *stat_tag() const override {
        return &p_->stat_tag;
    }

private:
    const prb_t *p_ = NULL;
};

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &mean,
        dnn_mem_t &var, const dnn_mem_t &ss, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &d_dst,
        const dnn_mem_t &ss, dnn_mem_t &d_src, dnn_mem_t &d_ss);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

} // namespace lnorm

#endif
