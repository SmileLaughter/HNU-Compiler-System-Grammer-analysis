"""
语法分析器主程序
"""

import os
import sys
from typing import List, Optional
from src.core.grammar_parser import GrammarParser, SentenceParser
from src.core.first_follow import FirstFollowCalculator
from src.parsers.ll1_parser import LL1Parser
from src.parsers.lr0_parser import LR0Parser
from src.parsers.slr_parser import SLRParser
from src.parsers.lr1_parser import LR1Parser
from src.parsers.lalr_parser import LALRParser
from src.core.parser_interface import ParserInterface
from src.utils.output_formatter import OutputFormatter
from rich.console import Console
from rich.prompt import Prompt, Confirm


class ParserManager:
    """语法分析器管理器"""
    
    def __init__(self):
        """初始化管理器"""
        self.grammar = None
        self.sentence = None
        self.sentences = []  # 存储多个句子
        self.calculator = None
        self.formatter = OutputFormatter()
        self.console = Console()
        
        # 可用的分析器列表（方便后续扩展）
        self.available_parsers = {
            '1': ('LL(1)', LL1Parser),
            '2': ('LR(0)', LR0Parser),
            '3': ('SLR', SLRParser),
            '4': ('LR(1)', LR1Parser),
            '5': ('LALR(1)', LALRParser),
        }
        
        # 配置DFA生成模式
        self._configure_dfa_mode()
    
    def _configure_dfa_mode(self):
        """配置DFA生成模式"""
        from src.config.dfa_config import dfa_config
        import config
        
        # 检查配置文件中的设置
        mode = config.DETERMINISTIC_MODE_CONFIG
        
        if mode == config.ConfigMode.ALWAYS_YES:
            # 总是启用确定性模式
            dfa_config.enable_deterministic_mode()
            self.console.print("[green]✓ 已启用确定性模式 (根据配置文件)[/green]")
            return
        elif mode == config.ConfigMode.ALWAYS_NO:
            # 总是禁用确定性模式
            dfa_config.disable_deterministic_mode()
            self.console.print("[yellow]✓ 已启用非确定性模式 (根据配置文件)[/yellow]")
            return
            
        # 否则询问用户
        self.console.print("\n[bold yellow]DFA生成模式配置[/bold yellow]")
        self.console.print("  [1] 确定性模式 - 每次运行生成完全相同的DFA（推荐）")
        self.console.print("  [2] 非确定性模式 - 生成同构但可能不同的DFA（更快）")
        
        choice = Prompt.ask(
            "\n请选择DFA生成模式",
            choices=["1", "2"],
            default="1"
        )
        
        if choice == "1":
            dfa_config.enable_deterministic_mode()
            self.console.print("[green]✓ 已启用确定性模式[/green]")
        else:
            dfa_config.disable_deterministic_mode()
            self.console.print("[yellow]✓ 已启用非确定性模式[/yellow]")
    
    def run(self):
        """运行主程序"""
        self.console.print(f"\n[bold cyan]{'=' * 40}[/bold cyan]")
        self.console.print("[bold cyan]语法分析器系统[/bold cyan]")
        self.console.print(f"[bold cyan]{'=' * 40}[/bold cyan]\n")
        
        # 步骤1：加载文法
        if not self.load_grammar():
            return
        
        # 步骤2：计算并显示集合
        self.calculate_and_display_sets()
        
        # 步骤3：选择算法或操作
        action = self.select_action()
        if action is None:
            return
        
        # 步骤4：根据选择执行对应操作
        if action == '6':
            # 最左推导 - 自动使用 LL(1)
            self.generate_leftmost_derivation()
        elif action == '7':
            # 最右推导 - 自动使用 LR(1)
            self.generate_rightmost_derivation()
        elif action in ['1', '2', '3', '4', '5']:
            # 标准语法分析 - 使用用户选择的算法
            parser_map = {
                '1': ('LL(1)', LL1Parser),
                '2': ('LR(0)', LR0Parser),
                '3': ('SLR', SLRParser),
                '4': ('LR(1)', LR1Parser),
                '5': ('LALR(1)', LALRParser)
            }
            parser_name, parser_class = parser_map[action]
            parser = parser_class(self.grammar)
            self.formatter.print_success(f"已选择 {parser_name} 分析算法")
            
            # 构建分析器（可能会触发文法转换）
            if not self.build_parser(parser):
                # 构建失败，询问是否继续（对于 LR 系列可能有冲突）
                if isinstance(parser, (LR0Parser, SLRParser, LR1Parser, LALRParser)):
                    self.formatter.print_separator()
                    from rich.prompt import Confirm
                    continue_anyway = Confirm.ask(
                        "\n分析表有冲突，是否仍然尝试解析句子？（可能会失败）", 
                        default=False
                    )
                    if not continue_anyway:
                        return
                else:
                    return
            
            # 显示分析表（如果文法被转换了，需要用新文法重建分析器）
            self.formatter.print_separator()
            if isinstance(parser, LL1Parser):
                final_parser = LL1Parser(self.grammar)
                final_parser.build()
                final_parser.print_table()
            # LR 系列的分析表已经在 build_parser 中显示了，不需要重复显示
            
            # 执行标准语法分析
            self.perform_standard_parsing(parser)
    
    def perform_standard_parsing(self, parser: ParserInterface):
        """
        执行标准的语法分析（原有功能）
        :param parser: 分析器对象
        """
        # 如果文件中有句子，直接进行语法分析；否则跳过
        if self.sentences:
            # 使用文件中的句子，循环解析每一个
            if isinstance(parser, LL1Parser):
                final_parser = LL1Parser(self.grammar)
                final_parser.build()
                for i, sentence in enumerate(self.sentences, 1):
                    self.formatter.print_separator()
                    sentence_str = ' '.join(sentence) if sentence else '(空串)'
                    self.console.print(f"\n[bold cyan]解析句子 [{i}/{len(self.sentences)}]: {sentence_str}[/bold cyan]")
                    self.parse_sentence(final_parser, sentence)
            else:
                # LR 系列直接使用之前构建的 parser
                for i, sentence in enumerate(self.sentences, 1):
                    self.formatter.print_separator()
                    sentence_str = ' '.join(sentence) if sentence else '(空串)'
                    self.console.print(f"\n[bold cyan]解析句子 [{i}/{len(self.sentences)}]: {sentence_str}[/bold cyan]")
                    self.parse_sentence(parser, sentence)
        else:
            # 没有句子，跳过语法分析步骤
            self.formatter.print_separator()
            self.formatter.print_info("未找到待分析的句子，跳过语法分析步骤")
    def load_grammar(self) -> bool:
        """
        加载文法和句子（从同一个文件）
        文件格式：
        - 文法部分
        %%
        - 句子部分（每行一个句子）
        :return: 是否成功加载
        """
        self.formatter.print_info("请指定文法文件")
        
        # 获取input文件夹路径
        input_dir = os.path.join(os.path.dirname(__file__), 'input')
        
        # 列出input文件夹中的所有文件
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
            if files:
                self.console.print("\n可用的文法文件：")
                for i, file in enumerate(files, 1):
                    self.console.print(f"  [{i}] {file}")
                
                # 让用户选择或输入文件名
                choice = Prompt.ask("\n请选择文件编号或输入文件名", default=files[0])
                
                # 判断是编号还是文件名
                if choice.isdigit() and 1 <= int(choice) <= len(files):
                    filename = files[int(choice) - 1]
                else:
                    filename = choice
                
                filepath = os.path.join(input_dir, filename)
            else:
                filepath = Prompt.ask("请输入文法文件路径")
        else:
            filepath = Prompt.ask("请输入文法文件路径")
        
        # 加载文法和句子
        try:
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含%%分隔符
            if '%%' in content:
                # 分割文法和句子
                parts = content.split('%%', 1)
                grammar_content = parts[0].strip()
                sentences_content = parts[1].strip() if len(parts) > 1 else ""
                
                # 解析文法
                from io import StringIO
                grammar_lines = grammar_content.split('\n')
                self.grammar = GrammarParser.parse_from_lines(grammar_lines)
                
                # 解析句子（每行一个）
                self.sentences = []
                if sentences_content:
                    for line in sentences_content.split('\n'):
                        line = line.strip()
                        # 跳过空行和注释行
                        if line and not line.startswith('#'):
                            # 将句子分割成符号列表
                            # 如果有空格，按空格分割；否则按字符分割
                            if ' ' in line:
                                sentence = line.split()
                            else:
                                # 没有空格，按字符分割（每个字符是一个符号）
                                sentence = list(line)
                            self.sentences.append(sentence)
                
                self.formatter.print_success(f"成功加载文法：{filepath}")
                if self.sentences:
                    self.formatter.print_success(f"成功加载 {len(self.sentences)} 个句子")
            else:
                # 没有%%分隔符，只当作文法文件
                self.grammar = GrammarParser.parse_from_file(filepath)
                self.sentences = []
                self.formatter.print_success(f"成功加载文法：{filepath}")
                self.formatter.print_info("文件中未找到句子（没有%%分隔符）")
            
            # 显示文法信息
            self.formatter.print_separator()
            self.formatter.print_grammar(self.grammar)
            
            # 显示句子信息
            if self.sentences:
                self.console.print(f"\n待分析的句子 ({len(self.sentences)} 个):")
                for i, sentence in enumerate(self.sentences, 1):
                    sentence_str = ' '.join(sentence) if sentence else '(空串)'
                    self.console.print(f"  [{i}] {sentence_str}")
            
            return True
        except Exception as e:
            self.formatter.print_error(f"加载文法失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_and_display_sets(self):
        """计算并显示NULLABLE、FIRST、FOLLOW集"""
        self.formatter.print_separator()
        self.formatter.print_info("正在计算 NULLABLE、FIRST、FOLLOW 集...")
        
        # 创建计算器并计算所有集合
        # 这里使用include_dollar=True（默认），因为：
        # 1. 可以同时用于LL(1)和LR系列的显示
        # 2. LL(1)解析器会创建自己的calculator（include_dollar=False）
        # 3. 显示FOLLOW集时会自动过滤$（通过get_follow_set方法）
        self.calculator = FirstFollowCalculator(self.grammar, include_dollar=True)
        self.calculator.calculate_all()
        
        # 显示NULLABLE集
        self.formatter.print_nullable_set(self.calculator.get_nullable_set())
        
        # 显示非终结符的FIRST集
        self.formatter.print_first_sets(self.grammar, self.calculator)
        
        # 显示非终结符的FOLLOW集
        self.formatter.print_follow_sets(self.grammar, self.calculator)
        
        # 显示产生式的FIRST集（实际是SELECT集）
        self.formatter.print_production_first_sets(self.grammar, self.calculator)
    
    def select_parser(self) -> ParserInterface:
        """
        让用户选择分析算法（仅用于标准语法分析）
        :return: 分析器对象，如果取消则返回None
        """
        self.formatter.print_separator()
        self.formatter.print_info("请选择语法分析算法")
        
        # 显示可用的分析器
        self.console.print("\n可用的分析算法：")
        for key, (name, _) in self.available_parsers.items():
            self.console.print(f"  [{key}] {name}")
        
        # 让用户选择
        choice = Prompt.ask("\n请选择算法编号", choices=list(self.available_parsers.keys()), default='1')
        
        # 创建分析器
        parser_name, parser_class = self.available_parsers[choice]
        parser = parser_class(self.grammar)
        
        self.formatter.print_success(f"已选择 {parser_name} 分析算法")
        return parser
    
    def select_action(self) -> Optional[str]:
        """
        选择要执行的操作（算法或功能）
        :return: 操作编号，如果取消则返回 None
        """
        self.formatter.print_separator()
        self.formatter.print_info("请选择语法分析算法")
        
        # 显示可用的分析算法
        self.console.print("\n可用的分析算法：")
        self.console.print("  [1] LL(1)")
        self.console.print("  [2] LR(0)")
        self.console.print("  [3] SLR")
        self.console.print("  [4] LR(1)")
        self.console.print("  [5] LALR(1)")
        
        # 显示可用的操作
        self.console.print("可用的操作：")
        self.console.print("  [6] 生成最左推导并显示语法分析树（自动使用 LL(1) 算法）")
        self.console.print("  [7] 生成最右推导并显示语法分析树（自动使用 LR(1) 算法）")
        
        # 让用户选择
        choice = Prompt.ask("\n请选择编号", choices=['1', '2', '3', '4', '5', '6', '7'], default='1')
        
        return choice
    
    def generate_leftmost_derivation(self):
        """生成最左推导（使用通用回溯搜索算法）"""
        if not self.sentences:
            self.formatter.print_error("未找到待分析的句子")
            return
        
        self.formatter.print_separator()
        self.console.print("[bold cyan]生成最左推导（支持任意CFG文法）[/bold cyan]\n")
        
        # 直接使用原始文法，由 DerivationGenerator 内部使用 LR(1) 处理
        from src.utils.derivation_generator import DerivationGenerator
        generator = DerivationGenerator(self.grammar, self.console)
        
        for i, sentence in enumerate(self.sentences, 1):
            self.formatter.print_separator()
            sentence_str = ' '.join(sentence) if sentence else '(空串)'
            self.console.print(f"\n[bold cyan]句子 [{i}/{len(self.sentences)}]: {sentence_str}[/bold cyan]")
            
            # 生成最左推导
            success, steps = generator.generate_leftmost_derivation(sentence)
            
            if success:
                # 显示推导过程
                generator.print_derivation(steps, "最左")
                
                # 生成语法分析树
                self.formatter.print_separator()
                import os
                
                # 生成文件名
                sentence_filename = '_'.join(sentence) if sentence else 'empty'
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    sentence_filename = sentence_filename.replace(char, '_')
                if len(sentence_filename) > 50:
                    sentence_filename = sentence_filename[:50]
                
                output_dir = "output/derivation_tree"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"leftmost_{sentence_filename}")
                
                # 传入推导步骤来构建语法树
                generator.visualize_parse_tree(sentence, steps, output_path)
            else:
                self.console.print("[bold red]✗ 无法生成最左推导（句子不合法）[/bold red]")
    
    def generate_rightmost_derivation(self):
        """生成最右推导（使用 LR(1) 算法反向推导）"""
        if not self.sentences:
            self.formatter.print_error("未找到待分析的句子")
            return
        
        self.formatter.print_separator()
        self.console.print("[bold cyan]使用 LR(1) 算法生成最右推导[/bold cyan]\n")
        
        # 自动构建 LR(1) 分析器
        from src.parsers.lr1_parser import LR1Parser
        lr1_parser = LR1Parser(self.grammar)
        lr1_parser.build()
        
        if lr1_parser.has_conflicts:
            self.console.print("[bold yellow]⚠ LR(1) 分析表有冲突，但仍尝试生成推导[/bold yellow]")
        
        for i, sentence in enumerate(self.sentences, 1):
            self.formatter.print_separator()
            sentence_str = ' '.join(sentence) if sentence else '(空串)'
            self.console.print(f"\n[bold cyan]句子 [{i}/{len(self.sentences)}]: {sentence_str}[/bold cyan]")
            
            # 使用 LR(1) 解析
            success, parse_result = lr1_parser.parse(sentence)
            
            if success and isinstance(parse_result, list):
                # 从 LR 规约序列反推最右推导
                derivation_steps = self._extract_rightmost_from_lr(parse_result, sentence)
                
                if derivation_steps:
                    # 显示推导过程
                    self._print_rightmost_derivation(derivation_steps)
                    
                    # 生成语法分析树
                    self.formatter.print_separator()
                    import os
                    
                    # 生成文件名
                    sentence_filename = '_'.join(sentence) if sentence else 'empty'
                    invalid_chars = '<>:"/\\|?*'
                    for char in invalid_chars:
                        sentence_filename = sentence_filename.replace(char, '_')
                    if len(sentence_filename) > 50:
                        sentence_filename = sentence_filename[:50]
                    
                    output_dir = "output/derivation_tree"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"rightmost_{sentence_filename}")
                    
                    # 如果解析结果中有语法树，直接使用
                    if 'parse_tree' in parse_result[-1]:
                        from src.utils.tree_visualizer import ParseTreeVisualizer
                        parse_tree = parse_result[-1]['parse_tree']
                        visualizer = ParseTreeVisualizer()
                        
                        visualizer.visualize(parse_tree, output_path, sentence_str)
                        self.console.print(f"\n[bold green]✓ 语法树已保存到：{output_path}.png[/bold green]")
                else:
                    self.console.print("[bold red]✗ 无法从 LR 解析结果中提取推导序列[/bold red]")
            else:
                self.console.print("[bold red]✗ LR(1) 解析失败（句子不合法）[/bold red]")
    
    def _extract_rightmost_from_lr(self, parse_result: List, sentence: List[str]):
        """
        从 LR 解析步骤中提取最右推导序列
        LR 分析器的归约序列实际上是最右推导的逆序
        """
        from src.utils.derivation_generator import DerivationStep
        
        # 提取所有归约步骤（使用的产生式）
        reductions = []
        for step in parse_result:
            if 'action' in step:
                action = step['action']
                # 匹配 "用 ... 归约" 格式
                if '归约' in action:
                    # 从 action 中提取产生式
                    # 格式: "用 F → id 归约"
                    import re
                    match = re.search(r'用\s+(.+?)\s+→\s+(.+?)\s+归约', action)
                    if match:
                        left = match.group(1).strip()
                        right_str = match.group(2).strip()
                        
                        # 查找匹配的产生式
                        for prod in self.grammar.productions:
                            if prod.left == left:
                                right_match = ' '.join(prod.right) if prod.right else 'ε'
                                if right_str == right_match or (right_str == 'ε' and prod.is_epsilon()):
                                    reductions.append(prod)
                                    break
        
        if not reductions:
            return None
        
        # 反转归约序列得到最右推导
        derivation_steps = []
        current_form = [self.grammar.start_symbol]
        derivation_steps.append(DerivationStep(current_form.copy(), None, 0))
        
        # 反向应用产生式（最右推导）
        for production in reversed(reductions):
            # 找到最右的匹配非终结符
            rightmost_pos = -1
            for i in range(len(current_form) - 1, -1, -1):
                if current_form[i] == production.left:
                    rightmost_pos = i
                    break
            
            if rightmost_pos == -1:
                # 找不到，尝试找第一个
                for i, symbol in enumerate(current_form):
                    if symbol == production.left:
                        rightmost_pos = i
                        break
            
            if rightmost_pos != -1:
                # 替换
                new_form = current_form[:rightmost_pos]
                if not production.is_epsilon():
                    new_form.extend(production.right)
                new_form.extend(current_form[rightmost_pos + 1:])
                
                current_form = new_form
                derivation_steps.append(DerivationStep(current_form.copy(), production, rightmost_pos))
        
        return derivation_steps
    
    def _print_rightmost_derivation(self, steps):
        """打印最右推导过程"""
        from rich.table import Table
        
        self.console.print("\n[bold cyan]最右推导过程：[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("步骤", style="cyan", width=6)
        table.add_column("句型", style="yellow")
        table.add_column("使用的产生式", style="green")
        
        for i, step in enumerate(steps):
            sentential_form_str = ' '.join(step.sentential_form)
            
            if step.production is None:
                table.add_row(str(i), sentential_form_str, "(起始)")
            else:
                prod_index = self.grammar.productions.index(step.production)
                prod_str = f"({prod_index}) {step.production}"
                table.add_row(str(i), sentential_form_str, prod_str)
        
        self.console.print(table)
        
        final_form = ' '.join(steps[-1].sentential_form)
        self.console.print(f"\n[bold green]✓ 最终句子：{final_form}[/bold green]")
    
    def build_parser(self, parser: ParserInterface) -> bool:
        """
        构建分析器，如果失败则尝试自动转换文法（针对 LL(1)）
        :param parser: 分析器对象
        :return: 是否构建成功
        """
        self.formatter.print_info(f"正在构建 {parser.get_name()} 分析表...")
        
        # 如果是 LL(1) 分析器，使用其特殊的构建和转换逻辑
        if isinstance(parser, LL1Parser):
            return self._build_ll1_parser(parser)
        
        # 如果是 LR 系列分析器，显示 DFA
        if isinstance(parser, (LR0Parser, SLRParser, LR1Parser, LALRParser)):
            return self._build_lr_parser(parser)
        
        # 其他分析器使用标准构建流程
        if parser.build():
            self.formatter.print_success("分析表构建成功")
            return True
        else:
            self.formatter.print_error("分析表构建失败")
            return False
    
    def _build_ll1_parser(self, parser: LL1Parser) -> bool:
        """
        构建 LL(1) 分析器的特殊逻辑（包含冲突解决）
        :param parser: LL(1) 分析器对象
        :return: 是否构建成功
        """
        # 尝试直接构建
        success, _ = parser.build_with_transform(self.console)
        
        if success:
            self.formatter.print_success("分析表构建成功")
            return True
        
        # 构建失败，询问是否尝试自动转换
        self.formatter.print_separator()
        try_transform = Confirm.ask(
            "\n是否尝试自动转换文法以解决冲突？", 
            default=True
        )
        
        if not try_transform:
            return False
        
        # 尝试自动转换
        self.formatter.print_separator()
        transform_success, transformed_grammar, transformations = parser.try_auto_transform(self.console)
        
        # 显示转换结果
        self.formatter.print_separator()
        parser.show_transform_result(transform_success, transformed_grammar,transformations, self.console)
        
        # 如果转换成功，更新文法并重新计算集合
        if transform_success:
            self.grammar = transformed_grammar
            self.calculator = FirstFollowCalculator(self.grammar)
            self.calculator.calculate_all()
            return True
        
        return False
    
    def _build_lr_parser(self, parser) -> bool:
        """
        构建 LR 系列分析器的特殊逻辑（显示 DFA）
        :param parser: LR 分析器对象
        :return: 是否构建成功
        """
        # 构建分析器（即使有冲突也会构建完成）
        parser.build()
        
        # 显示 DFA
        self.formatter.print_separator()
        parser.print_dfa()
        
        # 显示分析表
        self.formatter.print_separator()
        parser.print_table()
        
        import config
        
        # 询问是否生成DFA图片
        self.formatter.print_separator()
        
        gen_image_mode = config.GENERATE_DFA_IMAGE_CONFIG
        generate_dfa_image = False
        
        if gen_image_mode == config.ConfigMode.ALWAYS_YES:
            generate_dfa_image = True
            self.console.print("[cyan]正在生成DFA图片 (根据配置文件)...[/cyan]")
        elif gen_image_mode == config.ConfigMode.ALWAYS_NO:
            generate_dfa_image = False
        else:
            generate_dfa_image = Confirm.ask("\n是否将DFA生成图片并保存？", default=True)
        
        if generate_dfa_image:
            self._save_dfa_image(parser)
        
        # 询问是否导出DFA数据为JSON
        export_json_mode = config.EXPORT_DFA_JSON_CONFIG
        export_dfa_json = False
        
        if export_json_mode == config.ConfigMode.ALWAYS_YES:
            export_dfa_json = True
            self.console.print("[cyan]正在导出DFA数据 (根据配置文件)...[/cyan]")
        elif export_json_mode == config.ConfigMode.ALWAYS_NO:
            export_dfa_json = False
        else:
            export_dfa_json = Confirm.ask("\n是否将DFA导出为JSON格式？", default=True)
        
        if export_dfa_json:
            self._export_dfa_json(parser)
        
        # 检查是否有冲突
        if parser.has_conflicts:
            self.formatter.print_error(f"{parser.get_name()} 分析表构建失败：存在冲突")
            return False
        else:
            self.formatter.print_success("分析表构建成功")
            return True
    
    def _save_dfa_image(self, parser):
        """
        保存LR自动机DFA图片
        :param parser: LR分析器对象
        """
        from src.utils.dfa_visualizer import LRDFAVisualizer
        
        try:
            # 创建可视化器
            visualizer = LRDFAVisualizer(parser.automaton, parser.get_name())
            
            # 生成图片
            output_path = visualizer.visualize()
            
            self.formatter.print_success(f"DFA图片已保存到: {output_path}")
        except Exception as e:
            self.formatter.print_error(f"生成DFA图片失败: {str(e)}")
    
    def _export_dfa_json(self, parser):
        """
        导出LR自动机DFA为JSON格式
        :param parser: LR分析器对象
        """
        from src.utils.dfa_exporter import DFAExporter
        
        try:
            # 创建导出器
            exporter = DFAExporter(parser.automaton, parser.get_name())
            
            # 导出为JSON
            output_path = exporter.export_to_json()
            
            self.formatter.print_success(f"DFA数据已导出到: {output_path}")
        except Exception as e:
            self.formatter.print_error(f"导出DFA数据失败: {str(e)}")
    
    def load_sentence(self) -> bool:
        """
        加载待分析句子
        :return: 是否成功加载
        """
        self.formatter.print_separator()
        self.formatter.print_info("请指定待分析的句子")
        
        # 获取input文件夹路径
        input_dir = os.path.join(os.path.dirname(__file__), 'input')
        
        # 询问是从文件读取还是手动输入
        from_file = Confirm.ask("是否从文件读取句子？", default=True)
        
        if from_file:
            # 列出input文件夹中的所有文件
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
                if files:
                    self.console.print("\n可用的句子文件：")
                    for i, file in enumerate(files, 1):
                        self.console.print(f"  [{i}] {file}")
                    
                    # 让用户选择或输入文件名
                    choice = Prompt.ask("\n请选择文件编号或输入文件名", default=files[0] if files else "")
                    
                    # 判断是编号还是文件名
                    if choice.isdigit() and 1 <= int(choice) <= len(files):
                        filename = files[int(choice) - 1]
                    else:
                        filename = choice
                    
                    filepath = os.path.join(input_dir, filename)
                else:
                    filepath = Prompt.ask("请输入句子文件路径")
            else:
                filepath = Prompt.ask("请输入句子文件路径")
            
            # 加载句子
            try:
                self.sentence = SentenceParser.parse_from_file(filepath)
                sentence_str = ' '.join(self.sentence) if self.sentence else '(空串)'
                self.formatter.print_success(f"成功加载句子：{sentence_str}")
                return True
            except Exception as e:
                self.formatter.print_error(f"加载句子失败: {e}")
                return False
        else:
            # 手动输入句子
            sentence_input = Prompt.ask("请输入句子（符号之间用空格分隔，空串直接回车）")
            
            if sentence_input.strip():
                self.sentence = sentence_input.split()
            else:
                self.sentence = []
            
            sentence_str = ' '.join(self.sentence) if self.sentence else '(空串)'
            self.formatter.print_success(f"已输入句子：{sentence_str}")
            return True
    
    def parse_sentence(self, parser: ParserInterface, sentence: List[str]):
        """
        解析句子
        :param parser: 分析器对象
        :param sentence: 句子（符号列表）
        """
        self.formatter.print_separator()
        sentence_str = ' '.join(sentence) if sentence else '(空串)'
        self.formatter.print_info(f"开始解析句子：{sentence_str}")
        
        # 执行解析
        success, result = parser.parse(sentence)
        
        # 显示解析过程
        if isinstance(result, list):
            self.formatter.print_parsing_steps(result, success)
        else:
            self.console.print(result)
        
        # 如果解析成功，生成并显示语法树
        if success and isinstance(result, list) and len(result) > 0:
            # 检查最后一步是否包含语法树
            last_step = result[-1]
            if 'parse_tree' in last_step:
                from src.utils.tree_visualizer import ParseTreeVisualizer
                import os
                
                parse_tree = last_step['parse_tree']
                visualizer = ParseTreeVisualizer()
                
                # 生成文件名（使用句子作为文件名）
                sentence_filename = '_'.join(sentence) if sentence else 'empty'
                # 替换非法文件名字符
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    sentence_filename = sentence_filename.replace(char, '_')
                # 限制文件名长度
                if len(sentence_filename) > 50:
                    sentence_filename = sentence_filename[:50]
                
                output_dir = "output/grammar_tree"
                output_path = os.path.join(output_dir, sentence_filename)
                
                # 可视化并保存
                visualizer.visualize(parse_tree, output_path, sentence_str)


def main():
    """主函数"""
    manager = ParserManager()
    manager.run()


if __name__ == '__main__':
    main()
