# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
FROM python:2.7.14

# Install dependencies
COPY requirements.txt /home/
RUN pip install numpy==1.11.3 && \
    pip install gensim==3.0.0 && \
    pip install -r /home/requirements.txt

# Copy context into container file system
COPY . /home/
WORKDIR /home/

# Use matplotib 'Agg' backend
ENV MPLBACKEND "Agg"

# Export results
VOLUME /home/charts/
