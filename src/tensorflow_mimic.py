
class Graph():

    def __init__(self):
        self.operations = list();
        self.placeholders = list();
        self.variables = list();

        return;

    def setASDefault(self):
        global default_graph;
        default_graph = self;
        return


class Operations():

    def __init__(self, inputnodes=[]):
        self.input_nodes = inputnodes;
        self.output_nodes = [];

        for nodes in inputnodes:
            nodes.output_nodes.append(self);

        default_graph.operations.append(self);

        return;

    def compute(self, operand_a, operand_b):
        pass;


class Placeholders():

    def __init__(self):
        self.value = [];
        default_graph.placeholders.append(self);

        return;

class Variables():

    def __init__(self, init_val= 0):
        self.value = init_val;

        default_graph.variables.append(self);

        return;

class Add(Operations):

    def __init__(self, node_a, node_b):
        Operations.__init__([node_a, node_b]);
        return

    def compute(self, operand_a, operand_b):
        self.input = [operand_a, operand_b];

        return operand_a+operand_b;


g = Graph();
g.setASDefault();

a = Add(1, 2);

print(a.compute(3, 4))